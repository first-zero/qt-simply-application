#include <QtGui>
#include <QtWidgets>
#include <QMainWindow>

#include "opencv2/highgui/highgui.hpp"    
#include "opencv2/opencv.hpp"    
#include <opencv2/core/core.hpp> 
#include "header/Gray.h"

using namespace cv;

Gray::Gray()
{
	imgchangeClass = new ImgChange;
}

Gray::~Gray()
{
}

QImage Gray::Bin(QImage src, int threshold)			// ��ֵ��
{
	Mat srcImg, dstImg,grayImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
    if(srcImg.channels() != 1)
        cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	else
		dstImg = srcImg.clone();
	cv::threshold(grayImg, dstImg, threshold, 255, THRESH_BINARY);
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Gray::Graylevel(QImage src)					// �Ҷ�ͼ��
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
	dstImg.create(srcImg.size(), srcImg.type());
	if (srcImg.channels() != 1)
        cvtColor(srcImg, dstImg, COLOR_BGR2GRAY);
	else
		dstImg = srcImg.clone();
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}


QImage Gray::Reverse(QImage src)								// ͼ��ת
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
	bitwise_xor(srcImg, Scalar(255), dstImg);					// ���
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}


QImage Gray::Linear(QImage src, int alpha, int beta)		// ���Ա任
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
	srcImg.convertTo(dstImg, -1, alpha/100.0, beta-100);		// matDst = alpha * matTmp + beta 
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Gray::Gamma(QImage src, int gamma)				// ٤��任(ָ���任)
{
	if (gamma < 0)
		return src;

	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
	
	Mat lookUpTable(1, 256, CV_8U);                                    // ���ұ�
	uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i) {
        // s = c * r^gamma
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma/100.0)*255.0);      // pow()���ݴ�����
    }

	LUT(srcImg, lookUpTable, dstImg);                                   // LUT 

	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Gray::Log(QImage src, int c)			// �����任
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
    int val = 1;

	Mat lookUpTable(1, 256, CV_8U);                                    // ���ұ�
	uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i) {
        // s = c*log(1+r)  -> s = c * log_v^(1 + v*r)  [s,v ��0~1��]
//        p[i] = saturate_cast<uchar>((c/100.0)*log(1 + i / 255.0)*255.0);
        p[i] = saturate_cast<uchar>((c/100.0)*log(1 + (val+1.0) * i / 255.0)/ log(val+1.0) * 255.0);

    }
	LUT(srcImg, lookUpTable, dstImg);                                   // LUT 
	
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Gray::Histeq(QImage src)								// ֱ��ͼ���⻯
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);

	if (srcImg.channels() != 1)
        cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
	else
		dstImg = srcImg.clone();
	equalizeHist(srcImg, dstImg);                 // ֱ��ͼ���⻯

	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}
