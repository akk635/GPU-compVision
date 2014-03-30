#include "stereo_projection_gui.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    StereoProjectionGUI w;
    w.show();

    return a.exec();
}
