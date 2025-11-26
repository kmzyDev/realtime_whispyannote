#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <string>
#include <algorithm>
#include <chrono>

static std::vector<BYTE> g_loopBuffer;
static std::vector<BYTE> g_micBuffer;
static std::vector<BYTE> g_returnLoopBuffer;
static std::vector<BYTE> g_returnMicBuffer;
static std::mutex g_loopBufferMutex;
static std::mutex g_micBufferMutex;
static std::atomic<bool> g_loopRunning(false);
static std::atomic<bool> g_micRunning(false);
static std::thread g_loopThread;
static std::thread g_micThread;
static size_t g_minLoopBufferSize = 16000 * sizeof(int16_t) * 5;
static size_t g_minMicBufferSize = 16000 * sizeof(int16_t) * 4;

static std::wstring g_loopId;
static std::wstring g_micId;

inline float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

void ConvertFloat32ToInt16Mono16k(const float* pFloatSamples, size_t frameCount, int numChannels, int inputSampleRate, std::vector<BYTE>& output) {
    double resampleRatio = 16000.0 / inputSampleRate;
    size_t outFrames = static_cast<size_t>(frameCount * resampleRatio);

    output.clear();
    output.reserve(outFrames * sizeof(int16_t));

    for (size_t i = 0; i < outFrames; ++i) {
        double srcIndex = i / resampleRatio;
        size_t index0 = static_cast<size_t>(srcIndex);
        size_t index1 = std::min(index0 + 1, frameCount - 1);
        float frac = static_cast<float>(srcIndex - index0);

        float mono0 = 0.0f, mono1 = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch) {
            mono0 += pFloatSamples[index0 * numChannels + ch];
            mono1 += pFloatSamples[index1 * numChannels + ch];
        }
        mono0 /= numChannels;
        mono1 /= numChannels;

        float interpolated = mono0 * (1.0f - frac) + mono1 * frac;
        interpolated = clamp(interpolated, -1.0f, 1.0f);
        int16_t sample = static_cast<int16_t>(interpolated * 32767.0f);

        output.push_back(sample & 0xFF);
        output.push_back((sample >> 8) & 0xFF);
    }
}

void LoopRecordingThread(std::wstring loopId) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IMMDeviceEnumerator* pDeviceEnum = nullptr;
    IMMDevice *pLoopDevice = nullptr;
    IAudioClient *pLoopClient = nullptr;
    IAudioCaptureClient *pLoopCaptureClient = nullptr;
    WAVEFORMATEX *pLoopFmt = nullptr;

    HANDLE loopEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&pDeviceEnum));
    pDeviceEnum->GetDevice(loopId.c_str(), &pLoopDevice);
    pDeviceEnum->Release();

    pLoopDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pLoopClient);

    pLoopClient->GetMixFormat(&pLoopFmt);

    pLoopClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK, 0x989680, 0, pLoopFmt, nullptr);

    pLoopClient->SetEventHandle(loopEvent);

    pLoopClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pLoopCaptureClient);

    pLoopClient->Start();


    while (g_loopRunning) {
        DWORD wait = WaitForSingleObject(loopEvent, 1000);
        if (wait == WAIT_TIMEOUT) continue;

        BYTE *loopData = nullptr;
        UINT32 loopFrames = 0;
        DWORD loopFlags = 0;

        if (FAILED(pLoopCaptureClient->GetBuffer(&loopData, &loopFrames, &loopFlags, nullptr, nullptr)))
            break;

        std::vector<BYTE> loop16;
        ConvertFloat32ToInt16Mono16k((float*)loopData, loopFrames, pLoopFmt->nChannels, pLoopFmt->nSamplesPerSec, loop16);

        {
            std::lock_guard<std::mutex> lock(g_loopBufferMutex);
            g_loopBuffer.insert(g_loopBuffer.end(), loop16.begin(), loop16.end());
        }

        pLoopCaptureClient->ReleaseBuffer(loopFrames);
    }

    pLoopClient->Stop();
    if (pLoopCaptureClient) pLoopCaptureClient->Release();
    if (pLoopClient) pLoopClient->Release();
    if (pLoopDevice) pLoopDevice->Release();
    if (pLoopFmt) CoTaskMemFree(pLoopFmt);
    CloseHandle(loopEvent);
    CoUninitialize();
}

void MicRecordingThread(std::wstring micId) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IMMDeviceEnumerator* pDeviceEnum = nullptr;
    IMMDevice *pMicDevice = nullptr;
    IAudioClient *pMicClient = nullptr;
    IAudioCaptureClient *pMicCaptureClient = nullptr;
    WAVEFORMATEX *pMicFmt = nullptr;

    HANDLE micEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&pDeviceEnum));
    pDeviceEnum->GetDevice(micId.c_str(), &pMicDevice);
    pDeviceEnum->Release();

    pMicDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pMicClient);

    pMicClient->GetMixFormat(&pMicFmt);

    pMicClient->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, 0x989680, 0, pMicFmt, nullptr);

    pMicClient->SetEventHandle(micEvent);

    pMicClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pMicCaptureClient);

    pMicClient->Start();


    while (g_micRunning) {
        DWORD wait = WaitForSingleObject(micEvent, 1000);
        if (wait == WAIT_TIMEOUT) continue;

        BYTE *micData = nullptr;
        UINT32 micFrames = 0;
        DWORD micFlags = 0;

        if (FAILED(pMicCaptureClient->GetBuffer(&micData, &micFrames, &micFlags, nullptr, nullptr)))
            break;

        std::vector<BYTE> mic16;
        ConvertFloat32ToInt16Mono16k((float*)micData, micFrames, pMicFmt->nChannels, pMicFmt->nSamplesPerSec, mic16);

        {
            std::lock_guard<std::mutex> lock(g_micBufferMutex);
            g_micBuffer.insert(g_micBuffer.end(), mic16.begin(), mic16.end());
        }

        pMicCaptureClient->ReleaseBuffer(micFrames);
    }

    pMicClient->Stop();
    if (pMicCaptureClient) pMicCaptureClient->Release();
    if (pMicClient) pMicClient->Release();
    if (pMicDevice) pMicDevice->Release();
    if (pMicFmt) CoTaskMemFree(pMicFmt);
    CloseHandle(micEvent);
    CoUninitialize();
}

extern "C" {
    __declspec(dllexport)
    int StartLoopRecording(const char* loopId) {
        if (g_loopRunning) return 1;
        
        int loopLen = MultiByteToWideChar(CP_UTF8, 0, loopId, -1, nullptr, 0);
        if (loopLen > 0) {
            std::vector<wchar_t> loopBuffer(loopLen);
            MultiByteToWideChar(CP_UTF8, 0, loopId, -1, loopBuffer.data(), loopLen);
            g_loopId.assign(loopBuffer.data(), loopLen - 1);
        }

        g_loopRunning = true;
        g_loopThread = std::thread(LoopRecordingThread, g_loopId);
        return 0;
    }

    __declspec(dllexport)
    int StartMicRecording(const char* micId) {
        if (g_micRunning) return 1;
        
        int micLen = MultiByteToWideChar(CP_UTF8, 0, micId, -1, nullptr, 0);
        if (micLen > 0) {
            std::vector<wchar_t> micBuffer(micLen);
            MultiByteToWideChar(CP_UTF8, 0, micId, -1, micBuffer.data(), micLen);
            g_micId.assign(micBuffer.data(), micLen - 1);
        }

        g_micRunning = true;
        g_micThread = std::thread(MicRecordingThread, g_micId);
        return 0;
    }

    __declspec(dllexport)
    void StopLoopRecording() {
        if (!g_loopRunning) return;
        g_loopRunning = false;
        if (g_loopThread.joinable()) g_loopThread.join();
    }

    __declspec(dllexport)
    void StopMicRecording() {
        if (!g_micRunning) return;
        g_micRunning = false;
        if (g_micThread.joinable()) g_micThread.join();
    }

    __declspec(dllexport)
    int GetLoopAudio(unsigned char** buffer, int* length) {
        std::lock_guard<std::mutex> lock(g_loopBufferMutex);
        if (g_loopBuffer.size() < g_minLoopBufferSize) {
            *buffer = nullptr;
            *length = 0;
            return 0;
        }
        g_returnLoopBuffer = std::move(g_loopBuffer);
        g_loopBuffer.clear();
        *buffer = g_returnLoopBuffer.data();
        *length = (int)g_returnLoopBuffer.size();
        return 1;
    }

    __declspec(dllexport)
    int GetMicAudio(unsigned char** buffer, int* length) {
        std::lock_guard<std::mutex> lock(g_micBufferMutex);
        if (g_micBuffer.size() < g_minMicBufferSize) {
            *buffer = nullptr;
            *length = 0;
            return 0;
        }
        g_returnMicBuffer = std::move(g_micBuffer);
        g_micBuffer.clear();
        *buffer = g_returnMicBuffer.data();
        *length = (int)g_returnMicBuffer.size();
        return 1;
    }
}
