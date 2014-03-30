#include "stereo_projection_gui.h"
#include "ui_stereo_projection_gui.h"

// to store last open directory
const QString DEFAULT_DIR_KEY("LAST_DIR");
QSettings MySettings;


StereoProjectionGUI::StereoProjectionGUI(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::StereoProjectionGUI) {
    ui->setupUi(this);
}


void StereoProjectionGUI::on_run_pressed() {
    // first get impl type selected
    int implType = 0;
    QObjectList impls = ui->implType->children();
    for(int i = 0; i < impls.length(); i++) if(dynamic_cast<QRadioButton *>(impls.at(i))->isChecked()) implType = i;

    // construct cmd line arguments for core application
    QString cmd = "./main -i_left " + ui->leftImagePath->text() + " -i_right " + ui->rightImagePath->text() + " -mu " + ui->mu->text() + " -sigma " + ui->sigma->text() + " -tau " + ui->tau->text() + " -nt " + ui->nt->text() + " -steps " + ui->steps->text() + " -impl " + QString::number(implType) + " -suppress_out";

    // show running status in depth image label and disable run button
    ui->depthImageView->clear();
    ui->statusBar->showMessage("The program is running ... (please wait, don't panic if screen freezes)");
    ui->run->setEnabled(false);

    // start process and wait until finish
    QProcess *process;
    process = new QProcess(this);
    process->start(cmd);
    process->waitForFinished();

    // show result image
    ui->depthImageView->setPixmap(QPixmap( "images/out/depth_map.png"));
    ui->depthImageView->show();

    // show finish message and re-enable run button
    ui->statusBar->showMessage("Running finished. You can run it again!");
    ui->run->setEnabled(true);
}


void StereoProjectionGUI::on_leftImage_pressed() {    
    // open file using dialog
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), MySettings.value(DEFAULT_DIR_KEY).toString());
    ui->leftImagePath->setText(fileName);

    // update last accessed dir
    if (!fileName.isEmpty()) {
            QDir CurrentDir;
            MySettings.setValue(DEFAULT_DIR_KEY, CurrentDir.absoluteFilePath(fileName));
    }

    // preview selected image
    ui->leftImageView->setPixmap(QPixmap(fileName));
    ui->leftImageView->show();
}


void StereoProjectionGUI::on_rightImage_pressed() {
    // open file using dialog
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), MySettings.value(DEFAULT_DIR_KEY).toString());
    ui->rightImagePath->setText(fileName);

    // update last accessed dir
    if (!fileName.isEmpty()) {
            QDir CurrentDir;
            MySettings.setValue(DEFAULT_DIR_KEY, CurrentDir.absoluteFilePath(fileName));
    }

    // preview selected image
    ui->rightImageView->setPixmap(QPixmap(fileName));
    ui->rightImageView->show();
}


StereoProjectionGUI::~StereoProjectionGUI() {
    delete ui;
}
