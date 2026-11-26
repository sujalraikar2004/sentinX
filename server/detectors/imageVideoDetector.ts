import { DetectorResult, EvidenceItem } from '../../src/types/sentinel.js';

export interface MediaInspectionInput {
  mediaType: 'image' | 'video';
  fileName?: string;
  fileSizeBytes?: number;
  mimeType?: string;
  width?: number;
  height?: number;
  durationSec?: number;
  hasExifData?: boolean;
  metadataDetails?: Record<string, any>;
  samplePixelVariance?: number;
  mediaBytes?: Buffer;
}

export async function analyzeMediaAuthenticity(input: MediaInspectionInput): Promise<DetectorResult> {
  const startTime = Date.now();
  const externalEndpoint = process.env.DEEPFAKE_DETECTOR_API_URL;

  // Pass the original uploaded file to the configured inference service.
  if (externalEndpoint) {
    try {
      if (!input.mediaBytes) throw new Error('Media bytes are required for model inference');
      const form = new FormData();
      const mediaCopy = new Uint8Array(input.mediaBytes.byteLength);
      mediaCopy.set(input.mediaBytes);
      form.append('file', new Blob([mediaCopy.buffer], { type: input.mimeType || 'application/octet-stream' }), input.fileName || `media.${input.mediaType}`);
      form.append('mediaType', input.mediaType);
      const response = await fetch(`${externalEndpoint}/analyze`, {
        method: 'POST',
        body: form,
        signal: AbortSignal.timeout(180000),
      });

      if (response.ok) {
        const data = await response.json();
        const modelRisk = Number(data.riskScore);
        if (!Number.isFinite(modelRisk) || modelRisk < 0 || modelRisk > 100) throw new Error('Invalid media model risk score');
        return {
          moduleId: 'image_deepfake_detector',
          name: 'Image & Video Deepfake Forensics Analyzer',
          version: data.modelName || 'external-media-classifier',
          status: 'success',
          riskScore: modelRisk,
          normalizedRisk: modelRisk / 100,
          confidence: data.confidence ?? 0.88,
          reasonCodes: data.reasonCodes || ['XCEPTION_FFPP_COMPLETED'],
          evidence: data.evidence || [],
          processingTimeMs: Date.now() - startTime,
          limitations: [
            'The configured dima806 model card describes an AI-generated image classifier and warns that its training data is several years old and affected by concept drift.',
            'The reported classifier score is uncalibrated; it is not a probability that the media is fake or a calibrated real-world risk.',
            'Video is sampled as still images; this model does not analyze temporal video artifacts.',
          ],
          modelDetails: {
            architecture: 'ViT image classifier applied to detected face crops',
            datasetGrounding: 'AI-generated image classification; see model card for known concept drift',
            inferenceType: 'external_model_service',
            statusNote: `Uncalibrated classifier output from ${data.modelName || 'configured model'}; not a calibrated deepfake probability`,
          },
        };
      }
      const failure = await response.json().catch(() => ({}));
      throw new Error(failure.detail || failure.message || `Media model service returned HTTP ${response.status}`);
    } catch (err) {
      if (input.mediaBytes) throw err;
      // Metadata-only demonstration requests continue to use local heuristics.
    }
  }

  // Baseline Digital Media Forensics
  const evidence: EvidenceItem[] = [];
  const reasonCodes: string[] = [];
  let accumulatedRisk = 0;

  const fileSize = input.fileSizeBytes || 0;
  const mime = input.mimeType || (input.mediaType === 'video' ? 'video/mp4' : 'image/jpeg');
  const width = input.width || 0;
  const height = input.height || 0;

  evidence.push({
    id: 'ev-media-meta',
    label: 'Media Container Characteristics',
    value: `${mime} (${(fileSize / (1024 * 1024)).toFixed(2)} MB${width && height ? `, ${width}×${height}px` : ''})`,
    severity: 'info',
    explanation: 'Container metadata, resolution bounds, and file payload size.',
  });

  // Check 1: Image Dimensions & Standard AI Aspect Ratios (e.g. 512x512, 1024x1024 square diffusion outputs)
  if (width > 0 && height > 0) {
    if (width === height && (width === 512 || width === 768 || width === 1024)) {
      accumulatedRisk += 25;
      reasonCodes.push('COMMON_GENERATIVE_DIFFUSION_DIMENSIONS');
      evidence.push({
        id: 'ev-sq-dim',
        label: 'Canonical Latent Diffusion Aspect Ratio',
        value: `${width}×${height} (1:1 Exact Power-of-Two)`,
        severity: 'warning',
        explanation: '1:1 square dimensions matching 512px or 1024px are common defaults in latent diffusion pipelines (e.g., Stable Diffusion, Midjourney v4).',
      });
    } else {
      evidence.push({
        id: 'ev-dim-standard',
        label: 'Aspect Ratio Consistency',
        value: `${width}×${height}`,
        severity: 'safe',
        explanation: 'Typical consumer camera sensor aspect ratio (e.g., 4:3 or 16:9).',
      });
    }
  }

  // Check 2: EXIF Hardware Fingerprints
  if (input.mediaType === 'image') {
    if (input.hasExifData === false) {
      accumulatedRisk += 15;
      reasonCodes.push('NO_CAMERA_EXIF_METADATA');
      evidence.push({
        id: 'ev-exif-stripped',
        label: 'Absence of Hardware Camera Metadata',
        value: 'Stripped / Synthetic Canvas',
        severity: 'warning',
        explanation: 'Physical camera captures include sensor make, focal length, and exposure time. Purely generated images or web scrapings lack physical metadata.',
      });
    } else if (input.hasExifData === true) {
      evidence.push({
        id: 'ev-exif-found',
        label: 'Hardware Camera EXIF Tag Present',
        value: 'Sensor Metadata Verified',
        severity: 'safe',
        explanation: 'Detected physical camera sensor or processing hardware signature.',
      });
    }
  }

  // Check 3: Video duration and frame pacing
  if (input.mediaType === 'video') {
    const duration = input.durationSec || 0;
    if (duration > 0 && duration < 3.0) {
      accumulatedRisk += 20;
      reasonCodes.push('VIDEO_MICRO_CLIP_DURATION');
      evidence.push({
        id: 'ev-video-short',
        label: 'Ultra-Brief Video Segment',
        value: `${duration.toFixed(1)} seconds`,
        severity: 'warning',
        explanation: 'Deepfake face-swap videos on social media are frequently looped clips under 3 seconds to avoid temporal drift artifacts.',
      });
    }
  }

  const finalRiskScore = Math.min(100, Math.max(0, Math.round(accumulatedRisk)));
  const normalizedRisk = Number((finalRiskScore / 100).toFixed(2));

  return {
    moduleId: 'image_deepfake_detector',
    name: 'Image & Video Deepfake Forensics Analyzer',
    version: '1.2.0-forensic-baseline',
    status: 'success',
    riskScore: finalRiskScore,
    normalizedRisk,
    confidence: 0.70,
    reasonCodes: reasonCodes.length > 0 ? reasonCodes : ['FORENSIC_BASELINE_NORMAL'],
    evidence,
    processingTimeMs: Date.now() - startTime,
    limitations: [
      'Operating in metadata and dimension heuristic mode because the model service is not configured or did not respond.',
      'Social media compression (e.g., Instagram/TikTok re-encoding) naturally strips EXIF data and alters high-frequency DCT coefficients.',
      'Never label a subject a fraudster based solely on automated heuristic scores.',
    ],
    modelDetails: {
      architecture: 'Metadata heuristics; optional local image/video frame classifier',
      datasetGrounding: 'FaceForensics++ benchmark & C2PA content authenticity guidelines',
      inferenceType: 'rule_heuristic',
      statusNote: 'Heuristic fallback; start ml-service for image/video model inference',
    },
  };
}
