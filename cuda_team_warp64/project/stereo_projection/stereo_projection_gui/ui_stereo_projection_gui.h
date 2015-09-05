/********************************************************************************
** Form generated from reading UI file 'stereo_projection_gui.ui'
**
** Created: Sun Mar 30 01:38:11 2014
**      by: Qt User Interface Compiler version 4.8.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_STEREO_PROJECTION_GUI_H
#define UI_STEREO_PROJECTION_GUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_StereoProjectionGUI
{
public:
    QAction *actionQuit;
    QAction *actionQuit_2;
    QWidget *centralWidget;
    QLineEdit *sigma;
    QPushButton *run;
    QLineEdit *tau;
    QLabel *sigmaLabel;
    QLabel *tauLabel;
    QPushButton *leftImage;
    QLabel *ntLabel;
    QLineEdit *nt;
    QLabel *stepsLabel;
    QLineEdit *steps;
    QPushButton *rightImage;
    QLineEdit *leftImagePath;
    QLineEdit *rightImagePath;
    QLineEdit *mu;
    QLabel *muLabel;
    QLabel *leftImageView;
    QLabel *rightImageView;
    QLabel *depthImageView;
    QGroupBox *implType;
    QRadioButton *impl0;
    QRadioButton *impl1;
    QRadioButton *impl2;
    QRadioButton *impl3;
    QMenuBar *menuBar;
    QMenu *menuStereo_Projection;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *StereoProjectionGUI)
    {
        if (StereoProjectionGUI->objectName().isEmpty())
            StereoProjectionGUI->setObjectName(QString::fromUtf8("StereoProjectionGUI"));
        StereoProjectionGUI->resize(1051, 384);
        actionQuit = new QAction(StereoProjectionGUI);
        actionQuit->setObjectName(QString::fromUtf8("actionQuit"));
        actionQuit_2 = new QAction(StereoProjectionGUI);
        actionQuit_2->setObjectName(QString::fromUtf8("actionQuit_2"));
        centralWidget = new QWidget(StereoProjectionGUI);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        sigma = new QLineEdit(centralWidget);
        sigma->setObjectName(QString::fromUtf8("sigma"));
        sigma->setGeometry(QRect(70, 50, 113, 23));
        run = new QPushButton(centralWidget);
        run->setObjectName(QString::fromUtf8("run"));
        run->setGeometry(QRect(950, 300, 80, 23));
        tau = new QLineEdit(centralWidget);
        tau->setObjectName(QString::fromUtf8("tau"));
        tau->setGeometry(QRect(70, 90, 113, 23));
        sigmaLabel = new QLabel(centralWidget);
        sigmaLabel->setObjectName(QString::fromUtf8("sigmaLabel"));
        sigmaLabel->setGeometry(QRect(20, 50, 50, 15));
        tauLabel = new QLabel(centralWidget);
        tauLabel->setObjectName(QString::fromUtf8("tauLabel"));
        tauLabel->setGeometry(QRect(20, 90, 41, 16));
        leftImage = new QPushButton(centralWidget);
        leftImage->setObjectName(QString::fromUtf8("leftImage"));
        leftImage->setGeometry(QRect(280, 20, 80, 23));
        ntLabel = new QLabel(centralWidget);
        ntLabel->setObjectName(QString::fromUtf8("ntLabel"));
        ntLabel->setGeometry(QRect(20, 130, 41, 16));
        nt = new QLineEdit(centralWidget);
        nt->setObjectName(QString::fromUtf8("nt"));
        nt->setGeometry(QRect(70, 130, 113, 23));
        stepsLabel = new QLabel(centralWidget);
        stepsLabel->setObjectName(QString::fromUtf8("stepsLabel"));
        stepsLabel->setGeometry(QRect(20, 170, 41, 16));
        steps = new QLineEdit(centralWidget);
        steps->setObjectName(QString::fromUtf8("steps"));
        steps->setGeometry(QRect(70, 170, 113, 23));
        rightImage = new QPushButton(centralWidget);
        rightImage->setObjectName(QString::fromUtf8("rightImage"));
        rightImage->setGeometry(QRect(560, 20, 80, 23));
        leftImagePath = new QLineEdit(centralWidget);
        leftImagePath->setObjectName(QString::fromUtf8("leftImagePath"));
        leftImagePath->setEnabled(true);
        leftImagePath->setGeometry(QRect(240, 50, 161, 23));
        leftImagePath->setReadOnly(true);
        rightImagePath = new QLineEdit(centralWidget);
        rightImagePath->setObjectName(QString::fromUtf8("rightImagePath"));
        rightImagePath->setEnabled(true);
        rightImagePath->setGeometry(QRect(510, 50, 181, 23));
        rightImagePath->setReadOnly(true);
        mu = new QLineEdit(centralWidget);
        mu->setObjectName(QString::fromUtf8("mu"));
        mu->setGeometry(QRect(70, 10, 113, 23));
        muLabel = new QLabel(centralWidget);
        muLabel->setObjectName(QString::fromUtf8("muLabel"));
        muLabel->setGeometry(QRect(20, 10, 50, 15));
        leftImageView = new QLabel(centralWidget);
        leftImageView->setObjectName(QString::fromUtf8("leftImageView"));
        leftImageView->setGeometry(QRect(200, 80, 251, 181));
        leftImageView->setScaledContents(true);
        rightImageView = new QLabel(centralWidget);
        rightImageView->setObjectName(QString::fromUtf8("rightImageView"));
        rightImageView->setGeometry(QRect(480, 80, 251, 181));
        rightImageView->setScaledContents(true);
        depthImageView = new QLabel(centralWidget);
        depthImageView->setObjectName(QString::fromUtf8("depthImageView"));
        depthImageView->setGeometry(QRect(760, 80, 251, 181));
        depthImageView->setScaledContents(true);
        implType = new QGroupBox(centralWidget);
        implType->setObjectName(QString::fromUtf8("implType"));
        implType->setGeometry(QRect(20, 210, 161, 101));
        implType->setCheckable(false);
        implType->setChecked(false);
        impl0 = new QRadioButton(implType);
        impl0->setObjectName(QString::fromUtf8("impl0"));
        impl0->setGeometry(QRect(50, 20, 41, 21));
        impl0->setChecked(true);
        impl1 = new QRadioButton(implType);
        impl1->setObjectName(QString::fromUtf8("impl1"));
        impl1->setGeometry(QRect(50, 40, 41, 21));
        impl2 = new QRadioButton(implType);
        impl2->setObjectName(QString::fromUtf8("impl2"));
        impl2->setGeometry(QRect(50, 60, 41, 21));
        impl3 = new QRadioButton(implType);
        impl3->setObjectName(QString::fromUtf8("impl3"));
        impl3->setGeometry(QRect(50, 80, 41, 21));
        impl3->setChecked(false);
        StereoProjectionGUI->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(StereoProjectionGUI);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1051, 20));
        menuStereo_Projection = new QMenu(menuBar);
        menuStereo_Projection->setObjectName(QString::fromUtf8("menuStereo_Projection"));
        StereoProjectionGUI->setMenuBar(menuBar);
        mainToolBar = new QToolBar(StereoProjectionGUI);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        StereoProjectionGUI->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(StereoProjectionGUI);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        StereoProjectionGUI->setStatusBar(statusBar);

        menuBar->addAction(menuStereo_Projection->menuAction());
        menuStereo_Projection->addAction(actionQuit_2);

        retranslateUi(StereoProjectionGUI);

        QMetaObject::connectSlotsByName(StereoProjectionGUI);
    } // setupUi

    void retranslateUi(QMainWindow *StereoProjectionGUI)
    {
        StereoProjectionGUI->setWindowTitle(QApplication::translate("StereoProjectionGUI", "Stereo Projection - Team:: Warp64", 0, QApplication::UnicodeUTF8));
        actionQuit->setText(QApplication::translate("StereoProjectionGUI", "Quit", 0, QApplication::UnicodeUTF8));
        actionQuit_2->setText(QApplication::translate("StereoProjectionGUI", "Quit", 0, QApplication::UnicodeUTF8));
        sigma->setText(QApplication::translate("StereoProjectionGUI", "0.5", 0, QApplication::UnicodeUTF8));
        run->setText(QApplication::translate("StereoProjectionGUI", "Run", 0, QApplication::UnicodeUTF8));
        tau->setText(QApplication::translate("StereoProjectionGUI", "0.167", 0, QApplication::UnicodeUTF8));
        sigmaLabel->setText(QApplication::translate("StereoProjectionGUI", "Sigma", 0, QApplication::UnicodeUTF8));
        tauLabel->setText(QApplication::translate("StereoProjectionGUI", "Tau", 0, QApplication::UnicodeUTF8));
        leftImage->setText(QApplication::translate("StereoProjectionGUI", "Left Image", 0, QApplication::UnicodeUTF8));
        ntLabel->setText(QApplication::translate("StereoProjectionGUI", "Nt", 0, QApplication::UnicodeUTF8));
        nt->setText(QApplication::translate("StereoProjectionGUI", "16", 0, QApplication::UnicodeUTF8));
        stepsLabel->setText(QApplication::translate("StereoProjectionGUI", "Steps", 0, QApplication::UnicodeUTF8));
        steps->setText(QApplication::translate("StereoProjectionGUI", "400", 0, QApplication::UnicodeUTF8));
        rightImage->setText(QApplication::translate("StereoProjectionGUI", "Right Image", 0, QApplication::UnicodeUTF8));
        leftImagePath->setText(QString());
        rightImagePath->setText(QString());
        mu->setText(QApplication::translate("StereoProjectionGUI", "5", 0, QApplication::UnicodeUTF8));
        muLabel->setText(QApplication::translate("StereoProjectionGUI", "Mu", 0, QApplication::UnicodeUTF8));
        leftImageView->setText(QApplication::translate("StereoProjectionGUI", "                  No image loaded", 0, QApplication::UnicodeUTF8));
        rightImageView->setText(QApplication::translate("StereoProjectionGUI", "                    No Image Loaded", 0, QApplication::UnicodeUTF8));
        depthImageView->setText(QApplication::translate("StereoProjectionGUI", "                   Press Run to get output!", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        implType->setToolTip(QApplication::translate("StereoProjectionGUI", "Hover over buttions for more info...", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        implType->setTitle(QApplication::translate("StereoProjectionGUI", "Implementation Type", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        impl0->setToolTip(QApplication::translate("StereoProjectionGUI", "3D Grid with Global Memory", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        impl0->setText(QApplication::translate("StereoProjectionGUI", "0", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        impl1->setToolTip(QApplication::translate("StereoProjectionGUI", "3D Grid with Texture Memory", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        impl1->setText(QApplication::translate("StereoProjectionGUI", "1", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        impl2->setToolTip(QApplication::translate("StereoProjectionGUI", "3D Grid with Pitch Allocation", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        impl2->setText(QApplication::translate("StereoProjectionGUI", "2", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        impl3->setToolTip(QApplication::translate("StereoProjectionGUI", "3D Grid with Shared Memory", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        impl3->setText(QApplication::translate("StereoProjectionGUI", "3", 0, QApplication::UnicodeUTF8));
        menuStereo_Projection->setTitle(QApplication::translate("StereoProjectionGUI", "Stereo Projection", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class StereoProjectionGUI: public Ui_StereoProjectionGUI {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_STEREO_PROJECTION_GUI_H
