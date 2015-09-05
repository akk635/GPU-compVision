#ifndef STEREO_PROJECTION_GUI_H
#define STEREO_PROJECTION_GUI_H

#include <QMainWindow>
#include <QFileDialog>
#include <QProcess>
#include <QString>
#include <QSettings>

namespace Ui {
class StereoProjectionGUI;
}

class StereoProjectionGUI : public QMainWindow {
    Q_OBJECT

public:
    explicit StereoProjectionGUI(QWidget *parent = 0);
    ~StereoProjectionGUI();

private slots:
    // run button runs the stereo projection application and loads result
    void on_run_pressed();

    //when left image loaded show
    void on_leftImage_pressed();

    // when right image loaded show
    void on_rightImage_pressed();

private:
    Ui::StereoProjectionGUI *ui;
};

#endif // STEREO_PROJECTION_GUI_H
