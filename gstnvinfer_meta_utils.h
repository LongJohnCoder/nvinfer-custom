/**
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include "gstnvinfer.h"
#include "gstnvinfer_impl.h"

void attach_metadata_detector (GstNvinfercustom * nvinfer, GstMiniObject * tensor_out_object,
        GstNvinfercustomFrame & frame, NvDsInferDetectionOutput & detection_output);

void attach_metadata_classifier (GstNvinfercustom * nvinfer, GstMiniObject * tensor_out_object,
        GstNvinfercustomFrame & frame, GstNvinfercustomObjectInfo & object_info);

void merge_classification_output (GstNvinfercustomObjectHistory & history,
    GstNvinfercustomObjectInfo  &new_result);

void attach_metadata_segmentation (GstNvinfercustom * nvinfer, GstMiniObject * tensor_out_object,
        GstNvinfercustomFrame & frame, NvDsInferSegmentationOutput & segmentation_output);

/* Attaches the raw tensor output to the GstBuffer as metadata. */
void attach_tensor_output_meta (GstNvinfercustom *nvinfer, GstMiniObject * tensor_out_object,
        GstNvinfercustomBatch *batch, NvDsInferContextBatchOutput *batch_output);
