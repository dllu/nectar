#include <INectaCamera.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

using namespace CAlkUSB3;
using namespace std;

const int rows = 2048;
const int cols = 4096;

void capture(INectaCamera &cam, int shutterspeed, int analog_gain, int cds_gain) {
    // Get list of connected cameras
    Array<String> camList = cam.GetCameraList();
    if (camList.Size() == 0) {
        cout << "No AlkUSB3 camera found" << endl;
        exit(1);
    }

    cam.SetCamera(0);
    cam.Init();

    cam.SetADCResolution(12);
    cerr << (int)cam.GetMaxADCResolution() << endl;
    cam.SetVideoMode(1);
    cam.SetColorCoding(ColorCoding::Raw16);
    cam.SetImageSizeX(cols);
    cam.SetImageSizeY(rows);
    cam.SetGain(analog_gain);
    cam.SetCDSGain(cds_gain);
    cam.SetShutter(shutterspeed);
    cam.SetLinePeriod(cam.GetMinLinePeriod());
    cam.SetPacketSize(min(16384U, cam.GetMaxPacketSize()));

    cout << "Starting acquisition..." << endl;
    // Start acquisition

    int n = 1000;

    cam.SetAcquire(true);
    for (int f = 0; f < n; f++) {
        // Get new frame
        stringstream ss;
        ss << "im" << setfill('0') << setw(4) << f << ".bin";
        // cam.GetImagePtr(true).Save(ss.str().c_str());
        auto raw_image = cam.GetRawData();
        ofstream fout;
        fout.open(ss.str(), ios::out | ios::binary);
        fout.write((char*)raw_image.Data(), raw_image.Size());
        fout.close();
    }

    // Stop acquisition
    cam.SetAcquire(false);
    cout << "Acquisition stopped" << endl;
}

int main(int argc, char *argv[]) {
    INectaCamera &cam = INectaCamera::Create();
    if (argc < 5) {
        cerr << "USAGE: capture analog_gain cds_gain shutterspeed" << endl;
        return 1;
    }

    int analog_gain = atoi(argv[1]);
    int cds_gain = atoi(argv[2]);
    int shutterspeed = atoi(argv[3]);

    // capture
    try {
        capture(cam, shutterspeed, analog_gain, cds_gain);
    } catch (const Exception &ex) {
        cout << "Exception " << ex.Name() << " occurred" << endl
            << ex.Message() << endl;
    }
    return 0;
}
