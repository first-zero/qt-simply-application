#include "header/MainWindow.h"
#include <QtWidgets/QApplication>
#include <opencv2\dnn.hpp>
using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ImageProcessing w;
    w.show();
    return a.exec();
}
