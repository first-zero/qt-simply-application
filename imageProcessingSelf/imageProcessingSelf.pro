#-------------------------------------------------
#
# Project created by QtCreator 2019-04-24T17:33:41
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = imageProcessingSelf
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
    cpp/CreateMenu.cpp \
    cpp/CustomWindow.cpp \
    cpp/Enhance.cpp \
    cpp/Geom.cpp \
    cpp/Gray.cpp \
    cpp/ImgChange.cpp \
    cpp/main.cpp \
    cpp/MainWindow.cpp \
    cpp/PaintWidget.cpp

HEADERS += \
    header/CreateMenu.h \
    header/CustomWindow.h \
    header/Enhance.h \
    header/Geom.h \
    header/Gray.h \
    header/ImgChange.h \
    header/MainWindow.h \
    header/PaintWidget.h \
    ui_ImageProcessingSelf.h

FORMS += \
    ImageProcessingSelf.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += E:\tool\opencv4.1.0\opencv\build\include\opencv2 \
E:\tool\opencv4.1.0\opencv\build\include\opencv \
E:\tool\opencv4.1.0\opencv\build\include


LIBS += -LE:\tool\opencv4.1.0\opencv\build\x64\vc15\lib \
-lopencv_world410d

#DISTFILES += \
#    ../torch/candy.t7 \
#    candy.t7


