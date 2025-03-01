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

void PaintWidget::setImage(QImage img)				// 读入图像
{
	image = img; 
	tempImage = image;
}

QImage PaintWidget::getImage()						// 外部获取图像
{
	if (image.isNull() != true)
		return image;
}

void PaintWidget::setShape(PaintWidget::shape t)	// 设置绘制类型
{
	type = t;
}

void PaintWidget::setPenWidth(int w)					// 设置画笔宽度
{
	penWidth = w;
}

void PaintWidget::setPenColor(QColor c)							// 设置画笔颜色
{
	penColor = c;
}

void PaintWidget::paintEvent(QPaintEvent *) {
	QPainter painter(this);
	if (isDrawing == true) {
		painter.drawImage(0, 0, tempImage);// 如果正在绘图，既鼠标点击或者正在移动，画在tempImage上  
	}
	else {
        painter.drawImage(0, 0, image);// 如果鼠标释放，将图保存在image上
	}
}

void PaintWidget::mousePressEvent(QMouseEvent *event) {
	if (event->button() == Qt::LeftButton) {
		lastPoint = event->pos();
		isDrawing = true;// 鼠标点击开始绘图，移动表示正在绘图  
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

	if (event->buttons() & Qt::LeftButton) {// 鼠标为左键且正在移动  
        isMoving = true;
		endPoint = event->pos();
        tempImage = image; // 用image覆盖缓冲区
		if (type == Pen) {// 如果工具为画笔，不用双缓冲直接画在画板上  
			paint(image);
		}
		else { // 否则用双缓冲绘图  
            paint(tempImage);
		}
	}
}



// 实现非铅笔工具的绘制
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

	apen.setWidth(penWidth);	// 设置画笔宽度
	apen.setColor(penColor);	// 设置画笔颜色

    //默认就是实线描绘的
//    dashes<<Qt::SquareCap;
//    apen.setDashPattern(dashes);
    p.setPen(apen);// 设置绘图工具画笔线条宽度为4
    p.setRenderHint(QPainter::Antialiasing, true); // 反走样

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

	switch (type) {					// 画图 
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

	update();// 重绘  
}

