# Copyright (C) 2020-2022 Intel Corporation
# Copyright (C) 2022-2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import io
import os
import zipfile
from datetime import datetime
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import cv2
import PIL.Image
import pytz
from django.conf import settings
from django.core.cache import caches
from rest_framework.exceptions import NotFound, ValidationError

from cvat.apps.engine.cloud_provider import (Credentials,
                                             db_storage_to_storage_instance,
                                             get_cloud_storage_instance)
from cvat.apps.engine.log import ServerLogManager
from cvat.apps.engine.media_extractors import (ImageDatasetManifestReader,
                                               Mpeg4ChunkWriter,
                                               Mpeg4CompressedChunkWriter,
                                               VideoDatasetManifestReader,
                                               ZipChunkWriter,
                                               ZipCompressedChunkWriter)
from cvat.apps.engine.mime_types import mimetypes
from cvat.apps.engine.models import (DataChoice, DimensionType, Job, Image,
                                     StorageChoice, CloudStorage)
from cvat.apps.engine.utils import md5_hash
from utils.dataset_manifest import ImageManifestManager

slogger = ServerLogManager(__name__)

class MediaCache:
    def __init__(self, dimension=DimensionType.DIM_2D):
        self._dimension = dimension
        self._cache = caches['media']

    def _get_or_set_cache_item(self, key, create_function):
        slogger.glob.info(f'Starting to get chunk from cache: key {key}')
        item = self._cache.get(key)
        slogger.glob.info(f'Ending to get chunk from cache: key {key}, is_cached {bool(item)}')
        if not item:
            slogger.glob.info(f'Starting to prepare chunk: key {key}')
            item = create_function()
            slogger.glob.info(f'Ending to prepare chunk: key {key}')
            if item[0]:
                self._cache.set(key, item)

        return item

    def get_task_chunk_data_with_mime(self, chunk_number, quality, db_data):
        item = self._get_or_set_cache_item(
            key=f'{db_data.id}_{chunk_number}_{quality}',
            create_function=lambda: self._prepare_task_chunk(db_data, quality, chunk_number),
        )

        return item

    def get_selective_job_chunk_data_with_mime(self, chunk_number, quality, job):
        item = self._get_or_set_cache_item(
            key=f'{job.id}_{chunk_number}_{quality}',
            create_function=lambda: self.prepare_selective_job_chunk(job, quality, chunk_number),
        )

        return item

    def get_local_preview_with_mime(self, frame_number, db_data):
        item = self._get_or_set_cache_item(
            key=f'data_{db_data.id}_{frame_number}_preview',
            create_function=lambda: self._prepare_local_preview(frame_number, db_data),
        )

        return item

    def get_cloud_preview_with_mime(
        self,
        db_storage: CloudStorage,
    ) -> Optional[Tuple[io.BytesIO, str]]:
        key = f'cloudstorage_{db_storage.id}_preview'
        return self._cache.get(key)

    def get_or_set_cloud_preview_with_mime(
        self,
        db_storage: CloudStorage,
    ) -> Tuple[io.BytesIO, str]:
        key = f'cloudstorage_{db_storage.id}_preview'

        item = self._get_or_set_cache_item(
            key, create_function=lambda: self._prepare_cloud_preview(db_storage)
        )

        return item

    def get_frame_context_images(self, db_data, frame_number):
        item = self._get_or_set_cache_item(
            key=f'context_image_{db_data.id}_{frame_number}',
            create_function=lambda: self._prepare_context_image(db_data, frame_number)
        )

        return item

    @staticmethod
    def _get_frame_provider_class():
        from cvat.apps.engine.frame_provider import \
            FrameProvider  # TODO: remove circular dependency
        return FrameProvider

    def _prepare_task_chunk(self, db_data, quality, chunk_number):
        FrameProvider = self._get_frame_provider_class()

        writer_classes = {
            FrameProvider.Quality.COMPRESSED : Mpeg4CompressedChunkWriter if db_data.compressed_chunk_type == DataChoice.VIDEO else ZipCompressedChunkWriter,
            FrameProvider.Quality.ORIGINAL : Mpeg4ChunkWriter if db_data.original_chunk_type == DataChoice.VIDEO else ZipChunkWriter,
        }

        image_quality = 100 if writer_classes[quality] in [Mpeg4ChunkWriter, ZipChunkWriter] else db_data.image_quality
        mime_type = 'video/mp4' if writer_classes[quality] in [Mpeg4ChunkWriter, Mpeg4CompressedChunkWriter] else 'application/zip'

        kwargs = {}
        if self._dimension == DimensionType.DIM_3D:
            kwargs["dimension"] = DimensionType.DIM_3D
        writer = writer_classes[quality](image_quality, **kwargs)

        images = []
        buff = BytesIO()
        upload_dir = {
                StorageChoice.LOCAL: db_data.get_upload_dirname(),
                StorageChoice.SHARE: settings.SHARE_ROOT,
                StorageChoice.CLOUD_STORAGE: db_data.get_upload_dirname(),
            }[db_data.storage]
        if hasattr(db_data, 'video'):
            source_path = os.path.join(upload_dir, db_data.video.path)

            reader = VideoDatasetManifestReader(manifest_path=db_data.get_manifest_path(),
                source_path=source_path, chunk_number=chunk_number,
                chunk_size=db_data.chunk_size, start=db_data.start_frame,
                stop=db_data.stop_frame, step=db_data.get_frame_step())
            for frame in reader:
                images.append((frame, source_path, None))
        else:
            reader = ImageDatasetManifestReader(manifest_path=db_data.get_manifest_path(),
                chunk_number=chunk_number, chunk_size=db_data.chunk_size,
                start=db_data.start_frame, stop=db_data.stop_frame,
                step=db_data.get_frame_step())
            if db_data.storage == StorageChoice.CLOUD_STORAGE:
                db_cloud_storage = db_data.cloud_storage
                assert db_cloud_storage, 'Cloud storage instance was deleted'
                credentials = Credentials()
                credentials.convert_from_db({
                    'type': db_cloud_storage.credentials_type,
                    'value': db_cloud_storage.credentials,
                })
                details = {
                    'resource': db_cloud_storage.resource,
                    'credentials': credentials,
                    'specific_attributes': db_cloud_storage.get_specific_attributes()
                }
                cloud_storage_instance = get_cloud_storage_instance(cloud_provider=db_cloud_storage.provider_type, **details)
                for item in reader:
                    file_name = f"{item['name']}{item['extension']}"
                    with NamedTemporaryFile(mode='w+b', prefix='cvat', suffix=file_name.replace(os.path.sep, '#'), delete=False) as temp_file:
                        source_path = temp_file.name
                        buf = cloud_storage_instance.download_fileobj(file_name)
                        temp_file.write(buf.getvalue())
                        temp_file.flush()
                        checksum = item.get('checksum', None)
                        if not checksum:
                            slogger.cloud_storage[db_cloud_storage.id].warning('A manifest file does not contain checksum for image {}'.format(item.get('name')))
                        if checksum and not md5_hash(source_path) == checksum:
                            slogger.cloud_storage[db_cloud_storage.id].warning('Hash sums of files {} do not match'.format(file_name))
                        images.append((source_path, source_path, None))
            else:
                for item in reader:
                    source_path = os.path.join(upload_dir, f"{item['name']}{item['extension']}")
                    images.append((source_path, source_path, None))
        writer.save_as_chunk(images, buff)
        buff.seek(0)
        if db_data.storage == StorageChoice.CLOUD_STORAGE:
            images = [image[0] for image in images if os.path.exists(image[0])]
            for image_path in images:
                os.remove(image_path)
        return buff, mime_type

    def prepare_selective_job_chunk(self, db_job: Job, quality, chunk_number: int):
        db_data = db_job.segment.task.data

        FrameProvider = self._get_frame_provider_class()
        frame_provider = FrameProvider(db_data, self._dimension)

        frame_set = db_job.segment.frame_set
        frame_step = db_data.get_frame_step()
        chunk_frames = []

        writer = ZipCompressedChunkWriter(db_data.image_quality, dimension=self._dimension)
        dummy_frame = BytesIO()
        PIL.Image.new('RGB', (1, 1)).save(dummy_frame, writer.IMAGE_EXT)

        if hasattr(db_data, 'video'):
            frame_size = (db_data.video.width, db_data.video.height)
        else:
            frame_size = None

        for frame_idx in range(db_data.chunk_size):
            frame_idx = (
                db_data.start_frame + chunk_number * db_data.chunk_size + frame_idx * frame_step
            )
            if db_data.stop_frame < frame_idx:
                break

            frame_bytes = None

            if frame_idx in frame_set:
                frame_bytes = frame_provider.get_frame(frame_idx, quality=quality)[0]

                if frame_size is not None:
                    # Decoded video frames can have different size, restore the original one

                    frame = PIL.Image.open(frame_bytes)
                    if frame.size != frame_size:
                        frame = frame.resize(frame_size)

                    frame_bytes = BytesIO()
                    frame.save(frame_bytes, writer.IMAGE_EXT)
                    frame_bytes.seek(0)

            else:
                # Populate skipped frames with placeholder data,
                # this is required for video chunk decoding implementation in UI
                frame_bytes = BytesIO(dummy_frame.getvalue())

            if frame_bytes is not None:
                chunk_frames.append((frame_bytes, None, None))

        buff = BytesIO()
        writer.save_as_chunk(chunk_frames, buff, compress_frames=False,
            zip_compress_level=1 # these are likely to be many skips in SPECIFIC_FRAMES segments
        )
        buff.seek(0)

        return buff, 'application/zip'

    def _prepare_local_preview(self, frame_number, db_data):
        FrameProvider = self._get_frame_provider_class()
        frame_provider = FrameProvider(db_data, self._dimension)
        buff, mime_type = frame_provider.get_preview(frame_number)

        return buff, mime_type

    def _prepare_cloud_preview(self, db_storage):
        storage = db_storage_to_storage_instance(db_storage)
        if not db_storage.manifests.count():
            raise ValidationError('Cannot get the cloud storage preview. There is no manifest file')
        preview_path = None
        for manifest_model in db_storage.manifests.all():
            manifest_prefix = os.path.dirname(manifest_model.filename)
            full_manifest_path = os.path.join(db_storage.get_storage_dirname(), manifest_model.filename)
            if not os.path.exists(full_manifest_path) or \
                    datetime.utcfromtimestamp(os.path.getmtime(full_manifest_path)).replace(tzinfo=pytz.UTC) < storage.get_file_last_modified(manifest_model.filename):
                storage.download_file(manifest_model.filename, full_manifest_path)
            manifest = ImageManifestManager(
                os.path.join(db_storage.get_storage_dirname(), manifest_model.filename),
                db_storage.get_storage_dirname()
            )
            # need to update index
            manifest.set_index()
            if not len(manifest):
                continue
            preview_info = manifest[0]
            preview_filename = ''.join([preview_info['name'], preview_info['extension']])
            preview_path = os.path.join(manifest_prefix, preview_filename)
            break
        if not preview_path:
            msg = 'Cloud storage {} does not contain any images'.format(db_storage.pk)
            slogger.cloud_storage[db_storage.pk].info(msg)
            raise NotFound(msg)

        buff = storage.download_fileobj(preview_path)
        mime_type = mimetypes.guess_type(preview_path)[0]

        return buff, mime_type

    def _prepare_context_image(self, db_data, frame_number):
        zip_buffer = BytesIO()
        try:
            image = Image.objects.get(data_id=db_data.id, frame=frame_number)
        except Image.DoesNotExist:
            return None, None
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            if not image.related_files.count():
                return None, None
            common_path = os.path.commonpath(list(map(lambda x: str(x.path), image.related_files.all())))
            for i in image.related_files.all():
                path = os.path.realpath(str(i.path))
                name = os.path.relpath(str(i.path), common_path)
                image = cv2.imread(path)
                success, result = cv2.imencode('.JPEG', image)
                if not success:
                    raise Exception('Failed to encode image to ".jpeg" format')
                zip_file.writestr(f'{name}.jpg', result.tobytes())
        buff = zip_buffer.getvalue()
        mime_type = 'application/zip'
        return buff, mime_type
