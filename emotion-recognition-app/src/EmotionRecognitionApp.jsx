import { useState, useRef } from 'react';
import { Mic, Upload, Home, FileAudio, Volume2, AlertCircle } from 'lucide-react';

// Main App Component
export default function EmotionRecognitionApp() {
  const [activeTab, setActiveTab] = useState('home');
  const [emotionResult, setEmotionResult] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [audioFile, setAudioFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [recordedAudioUrl, setRecordedAudioUrl] = useState(null);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);

  const API_BASE_URL = 'http://localhost:8000';

  const processFileUpload = async (file) => {
    setIsLoading(true);
    setErrorMessage('');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE_URL}/emotion-recognition`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const result = await response.json();
      
      setEmotionResult({
        emotion: result.label,
        confidence: Math.round(result.probability * 100),
        source: 'upload'
      });
    } catch (error) {
      console.error("Error processing file:", error);
      setErrorMessage(`Failed to process file: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const processRecordedAudio = async (audioBlob) => {
    setIsLoading(true);
    setErrorMessage('');
    
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');
      
      const response = await fetch(`${API_BASE_URL}/emotion-recognition`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const result = await response.json();
      
      setEmotionResult({
        emotion: result.label,
        confidence: Math.round(result.probability * 100),
        source: 'recording'
      });
    } catch (error) {
      console.error("Error processing recorded audio:", error);
      setErrorMessage(`Failed to process recording: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      processFileUpload(file);
    }
  };

  // Convert recorded audio buffer to WAV format
  const convertToWav = (audioBuffer, numChannels, sampleRate) => {
    // Create a buffer view to write WAV header
    const buffer = new ArrayBuffer(44 + audioBuffer.length);
    const view = new DataView(buffer);
    
    // Write WAV header
    // "RIFF" chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + audioBuffer.length, true);
    writeString(view, 8, 'WAVE');
    
    // "fmt " sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // audio format (1 for PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * 2, true); // byte rate
    view.setUint16(32, numChannels * 2, true); // block align
    view.setUint16(34, 16, true); // bits per sample
    
    // "data" sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, audioBuffer.length, true);
    
    // Write audio data
    const offset = 44;
    for (let i = 0; i < audioBuffer.length; i++) {
      view.setInt8(offset + i, audioBuffer[i]);
    }
    
    return new Blob([view], { type: 'audio/wav' });
  };
  
  const writeString = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  const startRecording = async () => {
    setErrorMessage('');
    setRecordedAudioUrl(null);
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { 
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm' 
      });
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        try {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          
          const audioUrl = URL.createObjectURL(audioBlob);
          setRecordedAudioUrl(audioUrl);
          
          const arrayBuffer = await audioBlob.arrayBuffer();
          
          const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
          
          const wavChannels = audioBuffer.numberOfChannels;
          const wavSampleRate = audioBuffer.sampleRate;
          
          const leftChannel = audioBuffer.getChannelData(0);
          
          const samples = new Int16Array(leftChannel.length);
          for (let i = 0; i < leftChannel.length; i++) {
            // Convert float to int
            const s = Math.max(-1, Math.min(1, leftChannel[i]));
            samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          
          // Create WAV blob with correct format
          const wavBlob = new Blob([
            createWavHeader(samples.length, wavChannels, wavSampleRate),
            samples
          ], { type: 'audio/wav' });
          
          // Process the properly formatted WAV file
          processRecordedAudio(wavBlob);
        } catch (error) {
          console.error("Error processing audio:", error);
          setErrorMessage(`Failed to process audio: ${error.message}`);
        }
        
        // Stop all audio tracks
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error starting recording:", error);
      setErrorMessage(`Failed to access microphone: ${error.message}`);
    }
  };

  function createWavHeader(dataLength, numChannels = 1, sampleRate = 44100) {
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = dataLength * bytesPerSample;
    
    const buffer = new ArrayBuffer(44);
    const view = new DataView(buffer);
    
    // "RIFF" chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');
    
    // "fmt " sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, 1, true); // audio format (1 for PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bytesPerSample * 8, true); // bits per sample
    
    // "data" sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);
    
    return new Uint8Array(buffer);
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      happy: 'bg-amber-100 border-amber-400 text-amber-700',
      sad: 'bg-blue-100 border-blue-400 text-blue-700',
      angry: 'bg-red-100 border-red-400 text-red-700',
      neutral: 'bg-gray-100 border-gray-400 text-gray-700',
      fear: 'bg-purple-100 border-purple-400 text-purple-700',
      disgust: 'bg-green-100 border-green-400 text-green-700'
    };
    return colors[emotion] || 'bg-gray-100 border-gray-400 text-gray-700';
  };

  const getEmotionIcon = (emotion) => {
    switch(emotion) {
      case 'happy': return "😊";
      case 'sad': return "😢";
      case 'angry': return "😠";
      case 'neutral': return "😐";
      case 'fear': return "😨";
      case 'disgust': return "🤢";
      default: return "❓";
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md overflow-hidden">
        {/* Header */}
        <div className="bg-indigo-600 p-4 text-white">
          <h1 className="text-2xl font-bold text-center">Voice Emotion Recognition</h1>
        </div>

        {/* Navigation */}
        <div className="flex border-b">
          <button 
            onClick={() => setActiveTab('home')} 
            className={`flex items-center p-4 ${activeTab === 'home' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-gray-600'}`}
          >
            <Home className="mr-2" size={18} />
            Home
          </button>
          <button 
            onClick={() => setActiveTab('upload')} 
            className={`flex items-center p-4 ${activeTab === 'upload' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-gray-600'}`}
          >
            <Upload className="mr-2" size={18} />
            Upload Audio
          </button>
          <button 
            onClick={() => setActiveTab('record')} 
            className={`flex items-center p-4 ${activeTab === 'record' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-gray-600'}`}
          >
            <Mic className="mr-2" size={18} />
            Record Audio
          </button>
        </div>

        {/* Content Area */}
        <div className="p-6">
          {activeTab === 'home' && (
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-4">Welcome to the Emotion Recognition System</h2>
              <p className="text-gray-600 mb-6">
                This application uses a LSTM neural network to detect emotions from speech.
                You can upload an audio file or record your voice directly.
              </p>
              <div className="flex justify-center space-x-4">
                <button 
                  onClick={() => setActiveTab('upload')} 
                  className="bg-indigo-600 text-white px-6 py-2 rounded-lg flex items-center hover:bg-indigo-700"
                >
                  <Upload className="mr-2" size={18} />
                  Upload Audio
                </button>
                <button 
                  onClick={() => setActiveTab('record')} 
                  className="bg-indigo-600 text-white px-6 py-2 rounded-lg flex items-center hover:bg-indigo-700"
                >
                  <Mic className="mr-2" size={18} />
                  Record Audio
                </button>
              </div>
            </div>
          )}

          {activeTab === 'upload' && (
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-4">Upload Audio File</h2>
              <p className="text-gray-600 mb-6">
                Select an audio file to analyze the emotional content.
                Supported formats: WAV, MP3, OGG
              </p>
              
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 mb-6">
                <FileAudio size={48} className="mx-auto mb-4 text-gray-400" />
                <label className="bg-indigo-600 text-white px-6 py-2 rounded-lg cursor-pointer hover:bg-indigo-700">
                  Select Audio File
                  <input type="file" accept="audio/*" onChange={handleFileUpload} className="hidden" />
                </label>
                {audioFile && (
                  <p className="mt-4 text-gray-600">
                    Selected: {audioFile.name}
                  </p>
                )}
              </div>
            </div>
          )}

          {activeTab === 'record' && (
            <div className="text-center">
              <h2 className="text-xl font-semibold mb-4">Record Your Voice</h2>
              <p className="text-gray-600 mb-6">
                Click the button below to start recording your voice for emotion analysis.
              </p>
              
              <div className="flex justify-center mb-6">
                <button 
                  onClick={toggleRecording} 
                  className={`rounded-full p-6 flex items-center justify-center ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-indigo-600 hover:bg-indigo-700'}`}
                >
                  <Mic size={32} className="text-white" />
                </button>
              </div>
              
              <p className="text-gray-600">
                {isRecording ? "Recording... Click to stop and analyze" : "Click to start recording"}
              </p>

            {/* Audio preview */}
            {recordedAudioUrl && (
            <div className="mt-6 bg-slate-50 rounded-lg p-4 shadow-sm border border-slate-200">
                <p className="text-gray-700 mb-3 font-medium flex items-center">
                <FileAudio size={16} className="mr-2 text-indigo-500" />
                Recording preview:
                </p>
                <div className="bg-white rounded-md p-3 border border-slate-200">
                <audio 
                    src={recordedAudioUrl} 
                    controls 
                    className="w-full focus:outline-none" 
                    style={{
                    height: '40px',
                    borderRadius: '9999px'
                    }}
                />
                </div>
                <p className="text-xs text-gray-500 mt-2 text-center">
                Click play to listen to your recording before analysis
                </p>
            </div>
            )}
            </div>
          )}

          {/* Error Message */}
          {errorMessage && (
            <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
              <p>{errorMessage}</p>
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="mt-8 text-center">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-indigo-500 border-t-transparent"></div>
              <p className="mt-2 text-gray-600">Analyzing audio...</p>
            </div>
          )}

          {/* Results Section */}
          {emotionResult && !isLoading && (
            <div className="mt-8">
              <h3 className="text-lg font-semibold mb-4">Emotion Analysis Results</h3>
              <div className={`border rounded-lg p-6 ${getEmotionColor(emotionResult.emotion)}`}>
                <div className="flex justify-between items-center">
                  <div>
                    <p className="text-2xl font-bold mb-2">
                      {getEmotionIcon(emotionResult.emotion)} {emotionResult.emotion.charAt(0).toUpperCase() + emotionResult.emotion.slice(1)}
                    </p>
                    <p>Probability: {emotionResult.confidence}%</p>
                    <p className="text-sm mt-2">
                      Source: {emotionResult.source === 'upload' ? 'Uploaded file' : 'Voice recording'}
                    </p>
                  </div>
                  <Volume2 size={48} className="opacity-50" />
                </div>
              </div>
            </div>
          )}

          {/* Information Note */}
          <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start">
            <AlertCircle size={20} className="text-blue-500 mr-2 mt-1 flex-shrink-0" />
            <p className="text-sm text-blue-800">
              This application analyzes speech to detect emotions including happiness, sadness, anger, 
              fear, disgust, and neutral states. For accurate results, ensure clear audio with minimal 
              background noise.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
