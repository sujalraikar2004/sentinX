import express, { Request, Response, NextFunction } from 'express';
import { analyzeTextMessage, extractUrlsFromText } from './detectors/textDetector.js';
import { analyzeUrlSafety } from './detectors/urlDetector.js';
import { analyzeQrPayload } from './detectors/qrDetector.js';
import { analyzeAudioAuthenticity } from './detectors/audioDetector.js';
import { analyzeMediaAuthenticity } from './detectors/imageVideoDetector.js';
import { executeTrustFusion } from './fusion/trustFusionEngine.js';
import { generateGeminiExplanation } from './services/geminiExplanationService.js';
import { historyStore } from './services/historyStore.js';
import { AnalysisReport, DetectorResult, ModalityType, SystemStatusResponse } from '../src/types/sentinel.js';

const app = express();

// Middleware: JSON parser with payload limits
app.use(express.json({ limit: '40mb' }));
app.use(express.urlencoded({ extended: true, limit: '40mb' }));

// Start time for uptime metric
const serverStartTime = Date.now();

// 1. Health Endpoint
app.get('/api/health', (req: Request, res: Response) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: Math.round((Date.now() - serverStartTime) / 1000),
    service: 'SentinelX Core Engine',
  });
});

// 2. System Status Endpoint
app.get('/api/system/status', (req: Request, res: Response) => {
  const statusPayload: SystemStatusResponse = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptimeSeconds: Math.round((Date.now() - serverStartTime) / 1000),
    version: '3.0.0-academic-release',
    detectors: {
      text_nlp_detector: {
        status: 'operational',
        version: '1.4.0-scikitlearn-rules',
        description: 'NLP heuristic classification for urgency, credentials, coercion, and impersonation',
      },
      url_phishing_detector: {
        status: 'operational',
        version: '2.1.0-tldts-heuristics',
        description: 'Static lexical, homoglyph, and Public Suffix List phishing destination analyzer',
      },
      qr_analyzer: {
        status: 'operational',
        version: '1.2.0-jsqr-engine',
        description: 'Reed-Solomon matrix QR decoder with piped URL/URI scheme security evaluation',
      },
      audio_voice_spoof_detector: {
        status: 'operational',
        version: '1.1.0-heuristic-plus-optional-model',
        description: 'Audio feature heuristic, or uploaded waveform inference through the optional local model service',
      },
      image_deepfake_detector: {
        status: 'operational',
        version: '1.2.0-heuristic-plus-optional-model',
        description: 'Metadata heuristic, or image/video frame classification through the optional local model service',
      },
    },
    environment: {
      geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
      persistenceMode: 'local_storage',
      externalModelServiceUrl: process.env.VOICE_DETECTOR_API_URL || process.env.DEEPFAKE_DETECTOR_API_URL || null,
    },
  };
  res.json(statusPayload);
});

// 3. Detectors Status
app.get('/api/detectors/status', (req: Request, res: Response) => {
  res.json({
    timestamp: new Date().toISOString(),
    detectors: [
      { id: 'text_nlp_detector', name: 'Text & Message Threat Analyzer', status: 'operational', weight: 1.0 },
      { id: 'url_phishing_detector', name: 'URL & Phishing Domain Analyzer', status: 'operational', weight: 1.25 },
      { id: 'qr_analyzer', name: 'QR Code Security Analyzer', status: 'operational', weight: 1.1 },
      { id: 'audio_voice_spoof_detector', name: 'Voice Spoof & Synthetic-Speech Analyzer', status: 'operational', weight: 0.95 },
      { id: 'image_deepfake_detector', name: 'Image & Video Deepfake Forensics Analyzer', status: 'operational', weight: 0.95 },
    ],
  });
});

// Helper to construct and persist full AnalysisReport
async function finalizeAnalysis(params: {
  modalities: ModalityType[];
  detectors: Record<string, DetectorResult>;
  inputSummary: any;
  privacyMode?: boolean;
}): Promise<AnalysisReport> {
  const fusion = executeTrustFusion(params.detectors);
  const id = `sx-${Date.now().toString(36)}-${Math.random().toString(36).substring(2, 7)}`;

  // Collect all evidence items for Gemini summary
  const allEvidence = Object.values(params.detectors).flatMap(d => d.evidence);

  let geminiExplanation;
  try {
    geminiExplanation = await generateGeminiExplanation({
      trustRiskScore: fusion.trustRiskScore,
      riskBand: fusion.riskBand,
      modulesEvaluated: fusion.modulesEvaluated,
      primaryDrivers: fusion.primaryRiskDrivers,
      evidenceItems: allEvidence,
      inputSnippet: params.inputSummary.textSnippet || params.inputSummary.submittedUrl || params.inputSummary.qrPayload,
    });
  } catch {
    // Graceful fallback
  }

  const report: AnalysisReport = {
    id,
    timestamp: new Date().toISOString(),
    inputSummary: params.inputSummary,
    modalities: params.modalities,
    detectors: params.detectors,
    fusion,
    geminiExplanation,
    privacyMode: Boolean(params.privacyMode),
    systemVersion: '3.0.0-academic',
  };

  historyStore.saveReport(report);
  return report;
}

// 4. POST /api/analyze/text
app.post('/api/analyze/text', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { text, privacyMode } = req.body;
    if (typeof text !== 'string') {
      return res.status(400).json({ error: 'Field "text" must be a valid string.' });
    }

    const textResult = analyzeTextMessage(text);
    const detectors: Record<string, DetectorResult> = {
      text_nlp_detector: textResult,
    };

    // If text contains extracted URLs, also analyze embedded URLs!
    const extractedUrls = extractUrlsFromText(text);
    if (extractedUrls.length > 0) {
      const urlResult = analyzeUrlSafety(extractedUrls[0]);
      detectors.url_phishing_detector = urlResult;
    }

    const report = await finalizeAnalysis({
      modalities: ['text'],
      detectors,
      inputSummary: {
        textSnippet: text.slice(0, 160),
        textLength: text.length,
        submittedUrl: extractedUrls[0],
      },
      privacyMode,
    });

    res.json(report);
  } catch (err) {
    next(err);
  }
});

// 5. POST /api/analyze/url
app.post('/api/analyze/url', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { url, privacyMode } = req.body;
    if (typeof url !== 'string') {
      return res.status(400).json({ error: 'Field "url" must be a valid string.' });
    }

    const urlResult = analyzeUrlSafety(url);
    const detectors: Record<string, DetectorResult> = {
      url_phishing_detector: urlResult,
    };

    const report = await finalizeAnalysis({
      modalities: ['url'],
      detectors,
      inputSummary: {
        submittedUrl: url,
        normalizedUrl: url,
      },
      privacyMode,
    });

    res.json(report);
  } catch (err) {
    next(err);
  }
});

// 6. POST /api/analyze/qr
app.post('/api/analyze/qr', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { payload, width, height, privacyMode } = req.body;
    if (typeof payload !== 'string') {
      return res.status(400).json({ error: 'Field "payload" must be a valid string.' });
    }

    const qrResult = analyzeQrPayload(payload, width, height);
    const detectors: Record<string, DetectorResult> = {
      qr_analyzer: qrResult,
    };

    const report = await finalizeAnalysis({
      modalities: ['qr'],
      detectors,
      inputSummary: {
        qrPayload: payload,
      },
      privacyMode,
    });

    res.json(report);
  } catch (err) {
    next(err);
  }
});

// 7. POST /api/analyze/audio
app.post('/api/analyze/audio', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { fileName, fileSizeBytes, mimeType, durationSec, sampleRateHz, pcmBuffer, privacyMode } = req.body;

    const audioResult = await analyzeAudioAuthenticity({
      fileName,
      fileSizeBytes,
      mimeType,
      durationSec,
      sampleRateHz,
      pcmBuffer,
    });

    const detectors: Record<string, DetectorResult> = {
      audio_voice_spoof_detector: audioResult,
    };

    const report = await finalizeAnalysis({
      modalities: ['audio'],
      detectors,
      inputSummary: {
        fileName: fileName || 'audio_sample.wav',
        fileSize: fileSizeBytes,
        fileMimeType: mimeType || 'audio/wav',
        mediaDurationSec: durationSec,
      },
      privacyMode,
    });

    res.json(report);
  } catch (err) {
    next(err);
  }
});

// File-backed audio inference. The browser sends the original file as raw bytes;
// the detector service receives it as multipart form data for actual model inference.
app.post('/api/analyze/audio-file', express.raw({ type: 'application/octet-stream', limit: '15mb' }), async (req: Request, res: Response, next: NextFunction) => {
  try {
    if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
      return res.status(400).json({ error: 'An audio file is required.' });
    }
    const fileName = decodeURIComponent(String(req.headers['x-file-name'] || 'audio-sample'));
    const mimeType = String(req.headers['x-file-mime'] || 'application/octet-stream');
    const audioResult = await analyzeAudioAuthenticity({
      fileName,
      fileSizeBytes: req.body.length,
      mimeType,
      durationSec: Number(req.headers['x-duration-sec']) || undefined,
      sampleRateHz: Number(req.headers['x-sample-rate']) || undefined,
      audioBytes: req.body,
    });
    const report = await finalizeAnalysis({
      modalities: ['audio'],
      detectors: { audio_voice_spoof_detector: audioResult },
      inputSummary: { fileName, fileSize: req.body.length, fileMimeType: mimeType },
      privacyMode: String(req.headers['x-privacy-mode']) === 'true',
    });
    res.json(report);
  } catch (err) {
    next(err);
  }
});

// 8. POST /api/analyze/image
app.post('/api/analyze/image', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { fileName, fileSizeBytes, mimeType, width, height, hasExifData, privacyMode } = req.body;

    const imageResult = await analyzeMediaAuthenticity({
      mediaType: 'image',
      fileName,
      fileSizeBytes,
      mimeType,
      width,
      height,
      hasExifData,
    });

    const detectors: Record<string, DetectorResult> = {
      image_deepfake_detector: imageResult,
    };

    const report = await finalizeAnalysis({
      modalities: ['image'],
      detectors,
      inputSummary: {
        fileName: fileName || 'captured_image.jpg',
        fileSize: fileSizeBytes,
        fileMimeType: mimeType || 'image/jpeg',
      },
      privacyMode,
    });

    res.json(report);
  } catch (err) {
    next(err);
  }
});

// 9. POST /api/analyze/video
app.post('/api/analyze/video', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { fileName, fileSizeBytes, mimeType, width, height, durationSec, privacyMode } = req.body;

    const videoResult = await analyzeMediaAuthenticity({
      mediaType: 'video',
      fileName,
      fileSizeBytes,
      mimeType,
      width,
      height,
      durationSec,
    });

    const detectors: Record<string, DetectorResult> = {
      image_deepfake_detector: videoResult,
    };

    const report = await finalizeAnalysis({
      modalities: ['video'],
      detectors,
      inputSummary: {
        fileName: fileName || 'recorded_clip.mp4',
        fileSize: fileSizeBytes,
        fileMimeType: mimeType || 'video/mp4',
        mediaDurationSec: durationSec,
      },
      privacyMode,
    });

    res.json(report);
  } catch (err) {
    next(err);
  }
});

// File-backed image/video inference. This route is limited to the scanner's 25 MB upload cap.
app.post('/api/analyze/media-file', express.raw({ type: 'application/octet-stream', limit: '25mb' }), async (req: Request, res: Response, next: NextFunction) => {
  try {
    if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
      return res.status(400).json({ error: 'An image or video file is required.' });
    }
    const mediaType = String(req.headers['x-media-type']) === 'video' ? 'video' : 'image';
    const fileName = decodeURIComponent(String(req.headers['x-file-name'] || `upload.${mediaType}`));
    const mimeType = String(req.headers['x-file-mime'] || 'application/octet-stream');
    const mediaResult = await analyzeMediaAuthenticity({
      mediaType,
      fileName,
      fileSizeBytes: req.body.length,
      mimeType,
      width: Number(req.headers['x-width']) || undefined,
      height: Number(req.headers['x-height']) || undefined,
      durationSec: Number(req.headers['x-duration-sec']) || undefined,
      mediaBytes: req.body,
    });
    const report = await finalizeAnalysis({
      modalities: [mediaType],
      detectors: { image_deepfake_detector: mediaResult },
      inputSummary: { fileName, fileSize: req.body.length, fileMimeType: mimeType },
      privacyMode: String(req.headers['x-privacy-mode']) === 'true',
    });
    res.json(report);
  } catch (err) {
    next(err);
  }
});

// 10. POST /api/analyze/multimodal
app.post('/api/analyze/multimodal', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { text, url, qrPayload, audio: audioPayload, media: mediaPayload, privacyMode } = req.body;
    const audio = audioPayload && typeof audioPayload === 'object'
      ? { ...audioPayload, audioBytes: typeof audioPayload.fileBase64 === 'string' ? Buffer.from(audioPayload.fileBase64, 'base64') : undefined }
      : undefined;
    const media = mediaPayload && typeof mediaPayload === 'object'
      ? { ...mediaPayload, mediaBytes: typeof mediaPayload.fileBase64 === 'string' ? Buffer.from(mediaPayload.fileBase64, 'base64') : undefined }
      : undefined;
    const modalities: ModalityType[] = [];
    const detectors: Record<string, DetectorResult> = {};
    const inputSummary: any = {};

    // 1. Text modality
    if (text && typeof text === 'string' && text.trim().length > 0) {
      modalities.push('text');
      detectors.text_nlp_detector = analyzeTextMessage(text);
      inputSummary.textSnippet = text.slice(0, 160);
      inputSummary.textLength = text.length;

      // Check for inline URL
      const inlineUrls = extractUrlsFromText(text);
      if (inlineUrls.length > 0 && !url) {
        detectors.url_phishing_detector = analyzeUrlSafety(inlineUrls[0]);
        inputSummary.submittedUrl = inlineUrls[0];
        modalities.push('url');
      }
    }

    // 2. Explicit URL modality
    if (url && typeof url === 'string' && url.trim().length > 0) {
      if (!modalities.includes('url')) modalities.push('url');
      detectors.url_phishing_detector = analyzeUrlSafety(url);
      inputSummary.submittedUrl = url;
    }

    // 3. QR payload modality
    if (qrPayload && typeof qrPayload === 'string' && qrPayload.trim().length > 0) {
      modalities.push('qr');
      detectors.qr_analyzer = analyzeQrPayload(qrPayload);
      inputSummary.qrPayload = qrPayload;
    }

    // 4. Audio modality
    if (audio && typeof audio === 'object') {
      modalities.push('audio');
      detectors.audio_voice_spoof_detector = await analyzeAudioAuthenticity(audio);
      inputSummary.fileName = audio.fileName;
      inputSummary.fileSize = audio.fileSizeBytes;
      inputSummary.fileMimeType = audio.mimeType;
      inputSummary.mediaDurationSec = audio.durationSec;
    }

    // 5. Media (Image / Video) modality
    if (media && typeof media === 'object') {
      const type = media.mediaType === 'video' ? 'video' : 'image';
      modalities.push(type);
      detectors.image_deepfake_detector = await analyzeMediaAuthenticity(media);
      inputSummary.fileName = media.fileName;
      inputSummary.fileSize = media.fileSizeBytes;
      inputSummary.fileMimeType = media.mimeType;
      inputSummary.mediaDurationSec = media.durationSec;
    }

    if (modalities.length === 0) {
      return res.status(400).json({
        error: 'No valid modalities provided. Supply at least text, a URL, QR payload, or media attributes.',
      });
    }

    const report = await finalizeAnalysis({
      modalities,
      detectors,
      inputSummary,
      privacyMode,
    });

    res.json(report);
  } catch (err) {
    next(err);
  }
});

// 11. History Endpoints
app.get('/api/history', (req: Request, res: Response) => {
  const { search, modality, riskBand, limit, offset } = req.query;
  const result = historyStore.listReports({
    search: search ? String(search) : undefined,
    modality: modality ? String(modality) : undefined,
    riskBand: riskBand ? String(riskBand) : undefined,
    limit: limit ? Number(limit) : undefined,
    offset: offset ? Number(offset) : undefined,
  });
  res.json(result);
});

app.get('/api/history/:id', (req: Request, res: Response) => {
  const report = historyStore.getReport(req.params.id);
  if (!report) {
    return res.status(404).json({ error: 'Report not found' });
  }
  res.json(report);
});

app.delete('/api/history/:id', (req: Request, res: Response) => {
  const success = historyStore.deleteReport(req.params.id);
  if (!success) {
    return res.status(404).json({ error: 'Report not found' });
  }
  res.json({ success: true, deletedId: req.params.id });
});

app.delete('/api/history', (req: Request, res: Response) => {
  historyStore.clearAll();
  res.json({ success: true, message: 'All analysis history cleared.' });
});

// 12. Analytics Summary
app.get('/api/analytics/summary', (req: Request, res: Response) => {
  const summary = historyStore.getAnalyticsSummary();
  res.json(summary);
});

// Central Error Handler
app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  console.error('[SentinelX API Error]:', err?.message || err);
  res.status(err?.status || 500).json({
    error: 'Internal Server Error',
    message: err?.message || 'An unexpected error occurred during processing.',
    timestamp: new Date().toISOString(),
  });
});

export { app as apiApp };
