#include <QtGui>
#include <QtWidgets>
#include <QMainWindow>

#include "header/MainWindow.h"
#include "header/PaintWidget.h"

PaintWidget::PaintWidget(QWidget *parent) :
	QLabel(parent)
{
	isDrawing = false;
    isMoving = false;
    isCutting = false;
    type = Null;
//    QMainWindow* win;
//    win = (QMainWindow*)parent->parentWidget();

}
void PaintWidget::setI_MainWindow(ImageProcessing *p) {
    I_MainWindow = (ImageProcessing *)p;
}
void PaintWidget::saveImage(QImage Img) {
    if(I_MainWindow == 0)
        return;
//    QToolBar* I_ToolBar = new QToolBar ;

    I_MainWindow->ShowImage(Img, true);
    I_MainWindow->UpdateProp(Img);

}

void PaintWidget::setImage(QImage img)				// ����ͼ��
{
	image = img; 
	tempImage = image;
}

QImage PaintWidget::getImage()						// �ⲿ��ȡͼ��
{
	if (image.isNull() != true)
		return image;
}

void PaintWidget::setShape(PaintWidget::shape t)	// ���û�������
{
	type = t;
}

void PaintWidget::setPenWidth(int w)					// ���û��ʿ��
{
	penWidth = w;
}

void PaintWidget::setPenColor(QColor c)							// ���û�����ɫ
{
	penColor = c;
}

void PaintWidget::paintEvent(QPaintEvent *) {
	QPainter painter(this);
	if (isDrawing == true) {
		painter.drawImage(0, 0, tempImage);// ������ڻ�ͼ������������������ƶ�������tempImage��  
	}
	else {
        painter.drawImage(0, 0, image);// �������ͷţ���ͼ������image��
	}
}

void PaintWidget::mousePressEvent(QMouseEvent *event) {
	if (event->button() == Qt::LeftButton) {
		lastPoint = event->pos();
		isDrawing = true;// �������ʼ��ͼ���ƶ���ʾ���ڻ�ͼ  
	}
}
void PaintWidget::setCuttingStatus(bool b) {
    isCutting = b;
    type = Null;
}

bool PaintWidget::getCuttingStatus() {
    return isCutting;
}

void PaintWidget::mouseMoveEvent(QMouseEvent *event) {

	if (event->buttons() & Qt::LeftButton) {// ���Ϊ����������ƶ�  
        isMoving = true;
		endPoint = event->pos();
        tempImage = image; // ��image���ǻ�����
		if (type == Pen) {// �������Ϊ���ʣ�����˫����ֱ�ӻ��ڻ�����  
			paint(image);
		}
		else { // ������˫�����ͼ  
            paint(tempImage);
		}
	}
}



// ʵ�ַ�Ǧ�ʹ��ߵĻ���
void PaintWidget::mouseReleaseEvent(QMouseEvent *event) {
	isDrawing = false;
    if(!isMoving)
        return;
    isMoving = false;

    if(isCutting) {
        isCutting = false;
        int x1, y1, x2, y2, x, y;
        int width, height;
        x1 = lastPoint.x();
        y1 = lastPoint.y();
        x2 = endPoint.x();
        y2 = endPoint.y();

        x = x1 - x2 < 0? x1 : x2;
        width = abs(x1 - x2);
        y = y1 - y2 < 0? y1 : y2;
        height = abs(y1 - y2);

        if(width == 0 || height == 0)
            return;
//        Mat newImage = cv::Mat(new cv::Size(width, height), image.depth(), image.channels());

//        cv::Rect rect = cv::Rect(x, y, width, height);
        QImage newImage = image.copy(x, y, width, height);
        this->setImage(newImage);
        paint(image);
        saveImage(image);

        return;
    }
        if (type != Pen && type != Null) {
		paint(image);
//        qDebug() << "type:" <<type;
        saveImage(image);

	}
}

void PaintWidget::paint(QImage &theImage) {
	QPainter p(&theImage);
	QPen apen;	
    QVector<qreal> dashes;

	apen.setWidth(penWidth);	// ���û��ʿ��
	apen.setColor(penColor);	// ���û�����ɫ

    //Ĭ�Ͼ���ʵ������
//    dashes<<Qt::SquareCap;
//    apen.setDashPattern(dashes);
    p.setPen(apen);// ���û�ͼ���߻����������Ϊ4
    p.setRenderHint(QPainter::Antialiasing, true); // ������

	int x1, y1, x2, y2;
    x1 = lastPoint.x();
	y1 = lastPoint.y();
	x2 = endPoint.x();
	y2 = endPoint.y();

    if(isCutting&&isMoving) {
        qreal space = 4;
        dashes<<4 <<space <<4 <<space;
        apen.setDashPattern(dashes);
        p.setPen(apen);
        p.drawRect(x1, y1, x2 - x1, y2 - y1);
    }

	switch (type) {					// ��ͼ 
	case PaintWidget::Pen: 
	{
		p.drawLine(lastPoint, endPoint);
		lastPoint = endPoint;
		break;
	}
	case  PaintWidget::Line: 
	{
		p.drawLine(lastPoint, endPoint);
		break;
	}
	case PaintWidget::Ellipse: 
	{
		p.drawEllipse(x1, y1, x2 - x1, y2 - y1);
		break;
	}
	case PaintWidget::Circle: 
	{
		double length = (x2 - x1) > (y2 - y1) ? (x2 - x1) : (y2 - y1);
		p.drawEllipse(x1, y1, length, length);
		break;
	}
	case PaintWidget::Triangle: 
	{
		int top, buttom, left, right;
		top = (y1 < y2) ? y1 : y2;
		buttom = (y1 > y2) ? y1 : y2;
		left = (x1 < x2) ? x1 : x2;
		right = (x1 > x2) ? x1 : x2;

		if (y1 < y2)
		{
			QPoint points[3] = { QPoint(left,buttom),	QPoint(right,buttom),	QPoint((right + left) / 2,top) };
			p.drawPolygon(points, 3);
		}
		else
		{
			QPoint points[3] = { QPoint(left,top),	QPoint(right,top),	QPoint((left + right) / 2,buttom) };
			p.drawPolygon(points, 3);
		}
		break;
	}
	case PaintWidget::Rhombus: 
	{
		int top, buttom, left, right;
		top = (y1 < y2) ? y1 : y2;
		buttom = (y1 > y2) ? y1 : y2;
		left = (x1 < x2) ? x1 : x2;
		right = (x1 > x2) ? x1 : x2;

		QPoint points[4] = { 
			QPoint(left,(top + buttom) / 2),
			QPoint((left + right) / 2,buttom),
			QPoint(right,(top + buttom) / 2), 
			QPoint((left + right) / 2,top) };
		p.drawPolygon(points, 4);
		break;
	}
	case PaintWidget::Rect: 
	{
		p.drawRect(x1, y1, x2 - x1, y2 - y1);
		break;
	}
	case PaintWidget::Square: 
	{
		double length = (x2 - x1) > (y2 - y1) ? (x2 - x1) : (y2 - y1);
		p.drawRect(x1, y1, length, length);
		break;
	}
	case PaintWidget::Hexagon: 
	{
		QPoint points[6] = {
			QPoint(x1,y1),
			QPoint(x2,y1),
			QPoint((3 * x2 - x1) / 2,(y1 + y2) / 2),
			QPoint(x2,y2),
			QPoint(x1,y2),
			QPoint((3 * x1 - x2) / 2,(y1 + y2) / 2) };
		p.drawPolygon(points, 6);
		break;
	}
	case PaintWidget::Null:
	{
		;
	}
	default:
		break;
	}

	update();// �ػ�  
}

