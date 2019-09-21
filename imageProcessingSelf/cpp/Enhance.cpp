#include <QtGui>
#include <QtWidgets>
#include <QMainWindow>

#include "opencv2/highgui/highgui.hpp"    
#include "opencv2/opencv.hpp"    
#include <opencv2/core/core.hpp> 
#include <opencv2/dnn.hpp>
#include "header/Enhance.h"

using namespace cv;
using namespace std;

Enhance::Enhance()
{
	imgchangeClass = new ImgChange;
}

Enhance::~Enhance()
{
}

QImage Enhance::Normalized(QImage src,int kernel_length)								// 简单滤波
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
	blur(srcImg, dstImg, Size(kernel_length, kernel_length), Point(-1, -1));
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Gaussian(QImage src, int kernel_length)									// 高斯滤波
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
    // sigmaX x方向标准差， sigmaY ~
	GaussianBlur(srcImg, dstImg, Size(kernel_length, kernel_length), 0, 0);
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Median(QImage src, int kernel_length)									// 中值滤波
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);	
    // kernel_length 为奇数
	medianBlur(srcImg, dstImg, kernel_length);
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::HoughLine(QImage src, int threshold, double minLineLength, double maxLineGap)			// 线检测
{
	Mat srcImg, dstImg, cdstPImg;
	srcImg = imgchangeClass->QImage2cvMat(src);

    cv::Canny(srcImg, dstImg, 50, 200, 3);                // Canny算子边缘检测
	if (srcImg.channels() != 1)
		cvtColor(dstImg, cdstPImg, COLOR_GRAY2BGR);        // 转换灰度图像
	else
		cdstPImg = srcImg;

	vector<Vec4i> linesP;
	HoughLinesP(dstImg, linesP, 1, CV_PI / 180, threshold, minLineLength, maxLineGap);// 50,50,10
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(cdstPImg, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, LINE_AA);
	}

	QImage dst = imgchangeClass->cvMat2QImage(cdstPImg);
	return dst;

}

QImage Enhance::HoughCircle(QImage src, int minRadius, int maxRadius)		// 圆检测
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);

	Mat gray;
	if (srcImg.channels() != 1)
		cvtColor(srcImg, gray, COLOR_BGR2GRAY);
	else
		gray = srcImg;
	medianBlur(gray, gray, 5);              // 中值滤波，滤除噪声，避免错误检测

	vector<Vec3f> circles;
	HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 16, 100, 30, minRadius, maxRadius); // Hough圆检测,100, 30, 1, 30
	dstImg = srcImg.clone();

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center = Point(c[0], c[1]);
		circle(dstImg, center, 1, Scalar(0, 100, 100), 3, LINE_AA);                    // 画圆
		int radius = c[2];
		circle(dstImg, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
	}

	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Sobel(QImage src, int kernel_length)							// sobel
{
	Mat srcImg, dstImg, src_gray;
	srcImg = imgchangeClass->QImage2cvMat(src);

	GaussianBlur(srcImg, srcImg, Size(3, 3), 0, 0, BORDER_DEFAULT);     // 高斯模糊
	if (srcImg.channels() != 1)
		cvtColor(srcImg, src_gray, COLOR_BGR2GRAY);                        // 转换灰度图像
	else
		src_gray = srcImg;

	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;

	cv::Sobel(src_gray, grad_x, CV_16S, 1, 0, kernel_length, 1, 0, BORDER_DEFAULT);
	cv::Sobel(src_gray, grad_y, CV_16S, 0, 1, kernel_length, 1, 0, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x);            // 缩放，计算绝对值，并将结果转换为8位
	convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dstImg);

	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Laplacian(QImage src, int kernel_length)						// laplacian
{
	Mat srcImg, dstImg, src_gray;
	srcImg = imgchangeClass->QImage2cvMat(src);

	GaussianBlur(srcImg, srcImg, Size(3, 3), 0, 0, BORDER_DEFAULT);       // 高斯模糊

	if (srcImg.channels() != 1)
		cvtColor(srcImg, src_gray, COLOR_BGR2GRAY);                        // 转换灰度图像
	else
		src_gray = srcImg;

	Mat abs_dst;                                                    // 拉普拉斯二阶导数
	cv::Laplacian(src_gray, dstImg, CV_16S, kernel_length, 1, 0, BORDER_DEFAULT);

	convertScaleAbs(dstImg, dstImg);                                  // 绝对值8位
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Canny(QImage src, int kernel_length ,int lowThreshold)							// canny
{
	Mat srcImg, dstImg, src_gray, detected_edges;
	srcImg = imgchangeClass->QImage2cvMat(src);

	dstImg.create(srcImg.size(), srcImg.type());
    if (srcImg.channels() != 1)
		cvtColor(srcImg, src_gray, COLOR_BGR2GRAY);                        // 转换灰度图像
	else
		src_gray = srcImg;

    GaussianBlur(src_gray, detected_edges, Size(3, 3), 0, 0, BORDER_DEFAULT);       // 高斯模糊
//	blur(src_gray, detected_edges, Size(3, 3));     // 平均滤波平滑
    // th1 = 0.4*th2
	cv::Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * 3, kernel_length);
	dstImg = Scalar::all(0);
	srcImg.copyTo(dstImg, detected_edges);

	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

//----------------cv中取出	----------------开始

QImage Enhance::FastCycleGan(QImage src, String model)		// 风格迁移检测
{
    Mat srcImg, dstImg, src_gray, detected_edges;

    // 读取图片
    srcImg = imgchangeClass->QImage2cvMat(src);
//    srcImg = cv::imread("Test/test.jpg");

    dstImg.create(srcImg.size(), srcImg.type());
//        cv::imshow("sdf", srcImg);

    // 加载模型
    String modelPath = "../model/" + model +".t7";
    // "../model/candy.t7"
    dnn::Net net = cv::dnn::readNetFromTorch(modelPath);


    size_t h = srcImg.rows;// 行数(高度)
    size_t w = srcImg.cols;// 列数（宽度）

    Mat inputBlob;//转换为 (1,3,h,w) 的矩阵 即：(图像数,通道,高,宽)
                  //blobFromImage函数解释:

                  //第一个参数，InputArray image，表示输入的图像，可以是opencv的mat数据类型。
                  //第二个参数，scalefactor，这个参数很重要的，如果训练时，是归一化到0-1之间，那么这个参数就应该为0.00390625f （1/256），否则为1.0
                  //第三个参数，size，应该与训练时的输入图像尺寸保持一致。
                  //第四个参数，mean，这个主要在caffe中用到，caffe中经常会用到训练数据的均值。tf中貌似没有用到均值文件。
                  //第五个参数，swapRB，是否交换图像第1个通道和最后一个通道的顺序。
                  //第六个参数，crop，如果为true，就是裁剪图像，如果为false，就是等比例放缩图像。
                  //inputBlob= cv::dnn::blobFromImage(image, 1.0, Size(416, 416), Scalar(), false, true);//1/255.F
                  //inputBlob= cv::dnn::blobFromImage(image, 1.0, Size(416, 416*h/w), Scalar(103.939, 116.779, 123.680), false, true);//
    inputBlob = cv::dnn::blobFromImage(srcImg, 1.0, Size(w, h), Scalar(0.0, 0.0, 0.0), false, false);//

                                                                                                                // 进行计算
    net.setInput(inputBlob);
    Mat out = net.forward();

    Mat Styled;//4维转回3维
               //cv::dnn::imagesFromBlob(out, Styled);//由于这个函数会出错，已把它从dnn取出稍稍修改一下用
    imagesFromBlob(out, Styled);
    // 输出图片
    Styled /= 255;

    Mat uStyled;
    Styled.convertTo(uStyled, CV_8U, 255);//转换格式
//    cv::imshow("风格图像", uStyled);
    //    cv::imwrite("风格图像.jpg", uStyled);

    //    time1=((double)getTickCount()-time1)/getTickFrequency();//计算程序运行时间
    //    cout<<"此方法运行时间为："<<time1<<"秒"<<endl;//输出运行时间。


    QImage dst = imgchangeClass->cvMat2QImage(uStyled);
    return dst;

}

Mat Enhance::getPlane(const Mat &m, int n, int cn)
{
    const int cv_max_dim = 32;
    CV_Assert(m.dims > 2);
    int sz[cv_max_dim];
    for(int i = 2; i < m.dims; i++)
    {
        sz[i-2] = m.size.p[i];
    }
    // 2维 Size(width, height),  type,  data
    return Mat(m.dims - 2, sz, m.type(), (void*)m.ptr<float>(n, cn));
}

//仅用于单图
void Enhance::imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
{
    //blob 是浮点精度的4维矩阵
    //blob_[0] = 批量大小 = 图像数
    //blob_[1] = 通道数
    //blob_[2] = 高度
    //blob_[3] = 宽度
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    //images_.create(cv::Size(1, blob_.size[0]),blob_.depth() );//多图?
    images_.create(blob_.size[2],blob_.size[3],blob_.depth() );//创建一个图像


    std::vector<Mat> vectorOfChannels(blob_.size[1]);
    //for (int n = 0; n <  blob_.size[0]; ++n) //多个图
    {int n = 0;                                //只有一个图
        for (int c = 0; c < blob_.size[1]; ++c)
        {
            vectorOfChannels[c] = getPlane(blob_, n, c);
        }
        //cv::merge(vectorOfChannels, images_.getMatRef(n));//这里会出错，是前面的create的原因？
        cv::merge(vectorOfChannels, images_);//通道合并
    }
}
//----------------cv中取出	-----------------结束
