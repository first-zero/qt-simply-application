#include <QtGui>
#include <QtWidgets>
#include <QMainWindow>

#include "opencv2/highgui/highgui.hpp"    
#include "opencv2/opencv.hpp"    
#include <opencv2/core/core.hpp> 
#include "header/Geom.h"

using namespace cv;

Geom::Geom()
{
	imgchangeClass = new ImgChange;
}

Geom::~Geom()
{
}

QImage Geom::Resize(QImage src, int length, int width)		// 改变大小
{
	Mat matSrc, matDst;
//    Mat* test4 = new Mat(src.size().height(), src.size().width(),3);
//    test4->data = src.bits();
//    imshow("test4", *test4);
    matSrc = imgchangeClass->QImage2cvMat(src);
//        imshow("test1", matSrc);
	resize(matSrc, matDst, Size(length, width), 0, 0, INTER_LINEAR);// 线性插值
    QImage dst = imgchangeClass->cvMat2QImage(matDst);

//    Mat test3_1 = imgchangeClass->QImage2cvMat(dst);
//    imshow("test3-1", test3_1);

//    QImage dst_2 = imgchangeClass->cvMat2QImage(test3_1);
//    Mat test3_2 = imgchangeClass->QImage2cvMat(dst_2);
//    imshow("test3-2", test3_2);

//    QLabel* label2 = new QLabel("test2",0);
//    QPixmap mp;
//    mp = mp.fromImage(dst.rgbSwapped());
//    label2->setPixmap(mp);
//    label2->show();
	return dst;
}

QImage Geom::Enlarge_Reduce(QImage src, int times)			// 缩放
{
	Mat matSrc, matDst;
	matSrc = imgchangeClass->QImage2cvMat(src);
	if (times > 0)
	{
        resize(matSrc, matDst, Size(matSrc.cols * abs(times+1), matSrc.rows * abs(times+1)), 0, 0, INTER_LINEAR);
		QImage dst = imgchangeClass->cvMat2QImage(matDst);
		return dst;
	}
	else if (times < 0)
	{		
        resize(matSrc, matDst, Size(matSrc.cols / abs(times-1), matSrc.rows / abs(times-1)), 0, 0, INTER_AREA);
		QImage dst = imgchangeClass->cvMat2QImage(matDst);
		return dst;
	}
	else
	{
        matSrc = imgchangeClass->QImage2cvMat(src);
		return src;
	}
}

QImage Geom::Rotate(QImage src, int angle)							// 旋转
{
//    if(angle % 90 == 0)
//        return Rotate_fixed(src, angle);
    Mat matSrc, matDst,M;
	matSrc = imgchangeClass->QImage2cvMat(src);
	cv::Point2f center(matSrc.cols / 2, matSrc.rows / 2);
    cv::Rect bbox = cv::RotatedRect(center, matSrc.size(), angle).boundingRect();

    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1);

    rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
//    cv::Mat rot = cv::getRotationMatrix2D(Point2f(bbox.width, bbox.height/2),
//                                          angle, 1);

//    double a[2][3];
//    for(int i=0; i<2;i++)
//        for(int j=0; j<3;j++)
//            a[i][j] = rot.at<double>(i,j);

//    cv::warpAffine(matSrc, matDst, rot, bbox.size());
//    BORDER_REPLICATE
    cv::warpAffine(matSrc, matDst, rot, bbox.size(),
                   INTER_LINEAR, BORDER_CONSTANT , Scalar(0,0,0));
	QImage dst = imgchangeClass->cvMat2QImage(matDst);
	return dst;
}

QImage Geom::Rotate_fixed(QImage src, int angle)					// 旋转90，180，270
{
	Mat matSrc, matDst, M;
	matSrc = imgchangeClass->QImage2cvMat(src);
    M = getRotationMatrix2D(Point2i(matSrc.cols / 2, matSrc.rows / 2), angle, 1);

    int height = matSrc.rows;
    int width = matSrc.cols;
    if((angle - 90) % 180 == 0) {
            width = matSrc.rows;
            height = matSrc.cols;
            M.at<double>(0, 2) += (width - height) / 2.0;
            M.at<double>(1, 2) += (height - width) / 2.0;
    }
    warpAffine(matSrc, matDst, M, Size(width, height));
	QImage dst = imgchangeClass->cvMat2QImage(matDst);
	return dst;
}

QImage Geom::Flip(QImage src, int flipcode)							// 镜像
{
	Mat matSrc, matDst;
	matSrc = imgchangeClass->QImage2cvMat(src);
	flip(matSrc, matDst, flipcode);			// flipCode==0 垂直翻转（沿X轴翻转）,flipCode>0 水平翻转（沿Y轴翻转）
											// flipCode<0 水平垂直翻转（先沿X轴翻转，再沿Y轴翻转，等价于旋转180°）
	QImage dst = imgchangeClass->cvMat2QImage(matDst);
	return dst;
}

QImage Geom::Lean(QImage src, int x, int y)						// 倾斜
{
	Mat matSrc, matTmp, matDst;
	matSrc = imgchangeClass->QImage2cvMat(src);
	matTmp = Mat::zeros(matSrc.rows, matSrc.cols, matSrc.type());

	Mat map_x, map_y;
	Point2f src_point[3], tmp_point[3], x_point[3], y_point[3];
	double angleX = x / 180.0 * CV_PI ;
	double angleY = y / 180.0 * CV_PI;

	src_point[0] = Point2f(0, 0);	
	src_point[1] = Point2f(matSrc.cols, 0);
	src_point[2] = Point2f(0, matSrc.rows);

	x_point[0] = Point2f(matSrc.rows * tan(angleX), 0);
	x_point[1] = Point2f(matSrc.cols + matSrc.rows * tan(angleX), 0);
	x_point[2] = Point2f(0, matSrc.rows);
	
	map_x = getAffineTransform(src_point, x_point);
	warpAffine(matSrc, matTmp, map_x, Size(matSrc.cols + matSrc.rows * tan(angleX), matSrc.rows));

	tmp_point[0] = Point2f(0, 0);
	tmp_point[1] = Point2f(matTmp.cols, 0);
	tmp_point[2] = Point2f(0, matTmp.rows);

	y_point[0] = Point2f(0, 0);
	y_point[1] = Point2f(matTmp.cols, matTmp.cols * tan(angleY));
	y_point[2] = Point2f(0, matTmp.rows);

	map_y = getAffineTransform(tmp_point, y_point);
	warpAffine(matTmp, matDst, map_y, Size(matTmp.cols, matTmp.rows + matTmp.cols * tan(angleY)));

	QImage dst = imgchangeClass->cvMat2QImage(matDst);
	return dst;
}
