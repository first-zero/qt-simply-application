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

QImage Enhance::Normalized(QImage src,int kernel_length)								// ���˲�
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
	blur(srcImg, dstImg, Size(kernel_length, kernel_length), Point(-1, -1));
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Gaussian(QImage src, int kernel_length)									// ��˹�˲�
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);
    // sigmaX x�����׼� sigmaY ~
	GaussianBlur(srcImg, dstImg, Size(kernel_length, kernel_length), 0, 0);
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Median(QImage src, int kernel_length)									// ��ֵ�˲�
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);	
    // kernel_length Ϊ����
	medianBlur(srcImg, dstImg, kernel_length);
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::HoughLine(QImage src, int threshold, double minLineLength, double maxLineGap)			// �߼��
{
	Mat srcImg, dstImg, cdstPImg;
	srcImg = imgchangeClass->QImage2cvMat(src);

    cv::Canny(srcImg, dstImg, 50, 200, 3);                // Canny���ӱ�Ե���
	if (srcImg.channels() != 1)
		cvtColor(dstImg, cdstPImg, COLOR_GRAY2BGR);        // ת���Ҷ�ͼ��
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

QImage Enhance::HoughCircle(QImage src, int minRadius, int maxRadius)		// Բ���
{
	Mat srcImg, dstImg;
	srcImg = imgchangeClass->QImage2cvMat(src);

	Mat gray;
	if (srcImg.channels() != 1)
		cvtColor(srcImg, gray, COLOR_BGR2GRAY);
	else
		gray = srcImg;
	medianBlur(gray, gray, 5);              // ��ֵ�˲����˳����������������

	vector<Vec3f> circles;
	HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 16, 100, 30, minRadius, maxRadius); // HoughԲ���,100, 30, 1, 30
	dstImg = srcImg.clone();

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center = Point(c[0], c[1]);
		circle(dstImg, center, 1, Scalar(0, 100, 100), 3, LINE_AA);                    // ��Բ
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

	GaussianBlur(srcImg, srcImg, Size(3, 3), 0, 0, BORDER_DEFAULT);     // ��˹ģ��
	if (srcImg.channels() != 1)
		cvtColor(srcImg, src_gray, COLOR_BGR2GRAY);                        // ת���Ҷ�ͼ��
	else
		src_gray = srcImg;

	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;

	cv::Sobel(src_gray, grad_x, CV_16S, 1, 0, kernel_length, 1, 0, BORDER_DEFAULT);
	cv::Sobel(src_gray, grad_y, CV_16S, 0, 1, kernel_length, 1, 0, BORDER_DEFAULT);

	convertScaleAbs(grad_x, abs_grad_x);            // ���ţ��������ֵ���������ת��Ϊ8λ
	convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dstImg);

	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Laplacian(QImage src, int kernel_length)						// laplacian
{
	Mat srcImg, dstImg, src_gray;
	srcImg = imgchangeClass->QImage2cvMat(src);

	GaussianBlur(srcImg, srcImg, Size(3, 3), 0, 0, BORDER_DEFAULT);       // ��˹ģ��

	if (srcImg.channels() != 1)
		cvtColor(srcImg, src_gray, COLOR_BGR2GRAY);                        // ת���Ҷ�ͼ��
	else
		src_gray = srcImg;

	Mat abs_dst;                                                    // ������˹���׵���
	cv::Laplacian(src_gray, dstImg, CV_16S, kernel_length, 1, 0, BORDER_DEFAULT);

	convertScaleAbs(dstImg, dstImg);                                  // ����ֵ8λ
	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

QImage Enhance::Canny(QImage src, int kernel_length ,int lowThreshold)							// canny
{
	Mat srcImg, dstImg, src_gray, detected_edges;
	srcImg = imgchangeClass->QImage2cvMat(src);

	dstImg.create(srcImg.size(), srcImg.type());
    if (srcImg.channels() != 1)
		cvtColor(srcImg, src_gray, COLOR_BGR2GRAY);                        // ת���Ҷ�ͼ��
	else
		src_gray = srcImg;

    GaussianBlur(src_gray, detected_edges, Size(3, 3), 0, 0, BORDER_DEFAULT);       // ��˹ģ��
//	blur(src_gray, detected_edges, Size(3, 3));     // ƽ���˲�ƽ��
    // th1 = 0.4*th2
	cv::Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * 3, kernel_length);
	dstImg = Scalar::all(0);
	srcImg.copyTo(dstImg, detected_edges);

	QImage dst = imgchangeClass->cvMat2QImage(dstImg);
	return dst;
}

//----------------cv��ȡ��	----------------��ʼ

QImage Enhance::FastCycleGan(QImage src, String model)		// ���Ǩ�Ƽ��
{
    Mat srcImg, dstImg, src_gray, detected_edges;

    // ��ȡͼƬ
    srcImg = imgchangeClass->QImage2cvMat(src);
//    srcImg = cv::imread("Test/test.jpg");

    dstImg.create(srcImg.size(), srcImg.type());
//        cv::imshow("sdf", srcImg);

    // ����ģ��
    String modelPath = "../model/" + model +".t7";
    // "../model/candy.t7"
    dnn::Net net = cv::dnn::readNetFromTorch(modelPath);


    size_t h = srcImg.rows;// ����(�߶�)
    size_t w = srcImg.cols;// ��������ȣ�

    Mat inputBlob;//ת��Ϊ (1,3,h,w) �ľ��� ����(ͼ����,ͨ��,��,��)
                  //blobFromImage��������:

                  //��һ��������InputArray image����ʾ�����ͼ�񣬿�����opencv��mat�������͡�
                  //�ڶ���������scalefactor�������������Ҫ�ģ����ѵ��ʱ���ǹ�һ����0-1֮�䣬��ô���������Ӧ��Ϊ0.00390625f ��1/256��������Ϊ1.0
                  //������������size��Ӧ����ѵ��ʱ������ͼ��ߴ籣��һ�¡�
                  //���ĸ�������mean�������Ҫ��caffe���õ���caffe�о������õ�ѵ�����ݵľ�ֵ��tf��ò��û���õ���ֵ�ļ���
                  //�����������swapRB���Ƿ񽻻�ͼ���1��ͨ�������һ��ͨ����˳��
                  //������������crop�����Ϊtrue�����ǲü�ͼ�����Ϊfalse�����ǵȱ�������ͼ��
                  //inputBlob= cv::dnn::blobFromImage(image, 1.0, Size(416, 416), Scalar(), false, true);//1/255.F
                  //inputBlob= cv::dnn::blobFromImage(image, 1.0, Size(416, 416*h/w), Scalar(103.939, 116.779, 123.680), false, true);//
    inputBlob = cv::dnn::blobFromImage(srcImg, 1.0, Size(w, h), Scalar(0.0, 0.0, 0.0), false, false);//

                                                                                                                // ���м���
    net.setInput(inputBlob);
    Mat out = net.forward();

    Mat Styled;//4άת��3ά
               //cv::dnn::imagesFromBlob(out, Styled);//�����������������Ѱ�����dnnȡ�������޸�һ����
    imagesFromBlob(out, Styled);
    // ���ͼƬ
    Styled /= 255;

    Mat uStyled;
    Styled.convertTo(uStyled, CV_8U, 255);//ת����ʽ
//    cv::imshow("���ͼ��", uStyled);
    //    cv::imwrite("���ͼ��.jpg", uStyled);

    //    time1=((double)getTickCount()-time1)/getTickFrequency();//�����������ʱ��
    //    cout<<"�˷�������ʱ��Ϊ��"<<time1<<"��"<<endl;//�������ʱ�䡣


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
    // 2ά Size(width, height),  type,  data
    return Mat(m.dims - 2, sz, m.type(), (void*)m.ptr<float>(n, cn));
}

//�����ڵ�ͼ
void Enhance::imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
{
    //blob �Ǹ��㾫�ȵ�4ά����
    //blob_[0] = ������С = ͼ����
    //blob_[1] = ͨ����
    //blob_[2] = �߶�
    //blob_[3] = ���
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    //images_.create(cv::Size(1, blob_.size[0]),blob_.depth() );//��ͼ?
    images_.create(blob_.size[2],blob_.size[3],blob_.depth() );//����һ��ͼ��


    std::vector<Mat> vectorOfChannels(blob_.size[1]);
    //for (int n = 0; n <  blob_.size[0]; ++n) //���ͼ
    {int n = 0;                                //ֻ��һ��ͼ
        for (int c = 0; c < blob_.size[1]; ++c)
        {
            vectorOfChannels[c] = getPlane(blob_, n, c);
        }
        //cv::merge(vectorOfChannels, images_.getMatRef(n));//����������ǰ���create��ԭ��
        cv::merge(vectorOfChannels, images_);//ͨ���ϲ�
    }
}
//----------------cv��ȡ��	-----------------����
