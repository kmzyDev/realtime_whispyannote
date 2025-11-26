#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <iostream>
#include <vector>
#include <string>

EXTERN_C const PROPERTYKEY PKEY_Device_FriendlyName = {
    {0xa45c254e, 0xdf1c, 0x4efd, {0x80,0x20,0x67,0xd1,0x46,0xa8,0x50,0xe0}},
    14
};

struct AudioDevice {
    std::string name;
    std::string id;
    bool isActive;
};

std::vector<AudioDevice> renderDevices;

void EnumerateDevices(IMMDeviceEnumerator* pDeviceEnum, EDataFlow dataFlow, std::vector<AudioDevice>& deviceList) {
    deviceList.clear();
    
    IMMDeviceCollection* pDeviceCollection = nullptr;
    HRESULT hr = pDeviceEnum->EnumAudioEndpoints(
        dataFlow,
        DEVICE_STATE_ACTIVE,
        &pDeviceCollection
    );
    if (FAILED(hr)) {
        return;
    }

    UINT count = 0;
    pDeviceCollection->GetCount(&count);

    for (UINT i = 0; i < count; ++i) {
        IMMDevice* pDevice = nullptr;
        pDeviceCollection->Item(i, &pDevice);

        LPWSTR pDeviceId = nullptr;
        pDevice->GetId(&pDeviceId);

        IPropertyStore* pProps = nullptr;
        pDevice->OpenPropertyStore(STGM_READ, &pProps);

        PROPVARIANT varName;
        PropVariantInit(&varName);
        pProps->GetValue(PKEY_Device_FriendlyName, &varName);

        char nameBuffer[512];
        WideCharToMultiByte(CP_UTF8, 0, varName.pwszVal, -1, nameBuffer, sizeof(nameBuffer), nullptr, nullptr);
        char idBuffer[512];
        WideCharToMultiByte(CP_UTF8, 0, pDeviceId, -1, idBuffer, sizeof(idBuffer), nullptr, nullptr);
        
        DWORD state = 0;
        pDevice->GetState(&state);
        bool isActive = (state == DEVICE_STATE_ACTIVE);

        AudioDevice device;
        device.name = nameBuffer;
        device.id = idBuffer;
        device.isActive = isActive;
        deviceList.push_back(device);

        PropVariantClear(&varName);
        pProps->Release();
        CoTaskMemFree(pDeviceId);
        pDevice->Release();
    }

    pDeviceCollection->Release();
}
extern "C" {
    __declspec(dllexport)
    int GetDeviceCount() {
        return renderDevices.size();
    }

    __declspec(dllexport)
    const char* GetDeviceName(int index) {
        if (index < 0 || index >= static_cast<int>(renderDevices.size())) return nullptr;
        return renderDevices[index].name.c_str();
    }

    __declspec(dllexport)
    const char* GetDeviceId(int index) {
        if (index < 0 || index >= static_cast<int>(renderDevices.size())) return nullptr;
        return renderDevices[index].id.c_str();
    }

    __declspec(dllexport)
    int RefreshDevices(int dataFlow) {
        IMMDeviceEnumerator* pDeviceEnum = nullptr;
        HRESULT hr = CoCreateInstance(
            __uuidof(MMDeviceEnumerator),
            nullptr,
            CLSCTX_ALL,
            __uuidof(IMMDeviceEnumerator),
            (void**)&pDeviceEnum
        );
        if (FAILED(hr)) {
            return -1;
        }
        
        // 0: eRender, 1: eCapture
        EnumerateDevices(pDeviceEnum, static_cast<EDataFlow>(dataFlow), renderDevices);

        pDeviceEnum->Release();
        return 0;
    }
}
