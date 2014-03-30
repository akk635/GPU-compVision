/****************************************************************************
** Meta object code from reading C++ file 'stereo_projection_gui.h'
**
** Created: Sun Mar 30 01:38:13 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "stereo_projection_gui.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'stereo_projection_gui.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_StereoProjectionGUI[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      21,   20,   20,   20, 0x08,
      38,   20,   20,   20, 0x08,
      61,   20,   20,   20, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_StereoProjectionGUI[] = {
    "StereoProjectionGUI\0\0on_run_pressed()\0"
    "on_leftImage_pressed()\0on_rightImage_pressed()\0"
};

void StereoProjectionGUI::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        StereoProjectionGUI *_t = static_cast<StereoProjectionGUI *>(_o);
        switch (_id) {
        case 0: _t->on_run_pressed(); break;
        case 1: _t->on_leftImage_pressed(); break;
        case 2: _t->on_rightImage_pressed(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData StereoProjectionGUI::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject StereoProjectionGUI::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_StereoProjectionGUI,
      qt_meta_data_StereoProjectionGUI, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &StereoProjectionGUI::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *StereoProjectionGUI::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *StereoProjectionGUI::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_StereoProjectionGUI))
        return static_cast<void*>(const_cast< StereoProjectionGUI*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int StereoProjectionGUI::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
