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

#define EXPORT extern "C" __declspec(dllexport)

static std::vector<BYTE> g_buffer;
static std::vector<BYTE> g_returnBuffer;
static std::mutex g_buffer_mutex;
static std::atomic<bool> g_running(false);
static std::thread g_thread;
static size_t g_minBufferSize = 16000 * sizeof(int16_t) * 5;

static std::wstring g_loopId;

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

void CaptureThread(std::wstring loopId) {
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


    while (g_running) {
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
            std::lock_guard<std::mutex> lock(g_buffer_mutex);
            g_buffer.insert(g_buffer.end(), loop16.begin(), loop16.end());
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

EXPORT int StartCapture(const char* loopId) {
    if (g_running) return 1;
    
    int loopLen = MultiByteToWideChar(CP_UTF8, 0, loopId, -1, nullptr, 0);
    if (loopLen > 0) {
        std::vector<wchar_t> loopBuffer(loopLen);
        MultiByteToWideChar(CP_UTF8, 0, loopId, -1, loopBuffer.data(), loopLen);
        g_loopId.assign(loopBuffer.data(), loopLen - 1);
    }

    g_running = true;
    g_thread = std::thread(CaptureThread, g_loopId);
    return 0;
}

EXPORT void StopCapture() {
    if (!g_running) return;
    g_running = false;
    if (g_thread.joinable()) g_thread.join();
}

EXPORT int GetAudio(unsigned char** buffer, int* length) {
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    if (g_buffer.size() < g_minBufferSize) {
        *buffer = nullptr;
        *length = 0;
        return 0;
    }
    g_returnBuffer = std::move(g_buffer);
    g_buffer.clear();
    *buffer = g_returnBuffer.data();
    *length = (int)g_returnBuffer.size();
    return 1;
}
