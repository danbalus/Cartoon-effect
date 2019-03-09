// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
#include <math.h>
#include<algorithm>
#include <time.h>
using namespace std;

#define atW(x) (std::abs(x))
#define atN(y) (std::abs(y))
#define atE(x) (w-1-std::abs(w-1-(x)))
#define atS(y) (h-1-std::abs(h-1-(y)))

#define WEAK 128
#define STRONG 255



Mat conv(Mat src, Mat kernel) {

	Mat dst(src.rows, src.cols, CV_32FC1);

	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			float aux = 0;
			for (int k = 0; k < kernel.rows; k++)
			{
				for (int l = 0; l < kernel.cols; l++)
				{
					aux += kernel.at<float>(k, l) *(float)src.at<uchar>(i + k - 1, j + l - 1);
				}
			}
			dst.at<float>(i, j) = aux;
		}
	}
	return dst;

}



Mat negativul_imaginii2(Mat img)
{
	Mat rez(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			rez.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);//calculez negativul

	return rez;
}
Mat filtruMedian2(Mat src, int k) {


	Mat dst = src.clone();
	int w = 2 * k + 1;

	int d = w / 2;

	vector<uchar> vec;

	for (int i = k; i < src.rows - k; i++) {
		for (int j = k; j < src.cols - k; j++) {

			vec.clear();

			for (int ll = -d; ll <= d; ll++) {
				for (int lc = -d; lc <= d; lc++) {
					vec.push_back(src.at<uchar>(i + ll, j + lc));
				}
			}

			sort(vec.begin(), vec.end());

			dst.at<uchar>(i, j) = vec[w*w / 2];
		}
	}
	return dst;
}
float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));

}

void applyBilateralFilter(Mat source, Mat filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS) {
	double iFiltered = 0;
	double wP = 0;
	int neighbor_x = 0;
	int neighbor_y = 0;
	int half = diameter / 2;

	for (int i = 0; i < diameter; i++) {
		for (int j = 0; j < diameter; j++) {
			neighbor_x = x - (half - i);
			neighbor_y = y - (half - j);
			double gi = gaussian(source.at<uchar>(neighbor_x, neighbor_y) - source.at<uchar>(x, y), sigmaI);
			double gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
			double w = gi * gs;
			iFiltered = iFiltered + source.at<uchar>(neighbor_x, neighbor_y) * w;
			wP = wP + w;
		}
	}
	iFiltered = iFiltered / wP;
	filteredImage.at<double>(x, y) = iFiltered;


}


void apply_bilateral_filter(const cv::Mat_<double>& src, cv::Mat_<double>& dst, double sigmaS, double sigmaR, int tone = 256)
{
	assert(src.size() == dst.size());

	const int w = src.cols;
	const int h = src.rows;

	// generating spatial kernel
	int r = int(ceil(4.0*sigmaS));
	cv::Mat_<double> kernelS(1 + r, 1 + r);
	for (int v = 0; v <= r; ++v)
		for (int u = 0; u <= r; ++u)
			kernelS(v, u) = exp(-0.5*(u*u + v * v) / (sigmaS*sigmaS));

	// generating range kernel (discretized for fast computation)
	std::vector<double> kernelR(tone);
	for (int t = 0; t<tone; ++t)
		kernelR[t] = exp(-0.5*t*t / (sigmaR*sigmaR));

	// filtering
	for (int y = 0; y<h; ++y)
		for (int x = 0; x<w; ++x)
		{
			double p = src(y, x);
			int t = int(p);

			double numer = 1.0, denom = p; // (0,0)
			for (int u = 1; u <= r; ++u) // (u,0)
			{
				double p0 = src(y, atW(x - u));
				double p1 = src(y, atE(x + u));
				int t0 = int(p0);
				int t1 = int(p1);
				double wr0 = kernelR[abs(t0 - t)];
				double wr1 = kernelR[abs(t1 - t)];
				numer += kernelS(0, u)*(wr0 + wr1);
				denom += kernelS(0, u)*(wr0*p0 + wr1 * p1);
			}
			for (int v = 1; v <= r; ++v) // (0,v)
			{
				double p0 = src(atN(y - v), x);
				double p1 = src(atS(y + v), x);
				int t0 = int(p0);
				int t1 = int(p1);
				double wr0 = kernelR[abs(t0 - t)];
				double wr1 = kernelR[abs(t1 - t)];
				numer += kernelS(v, 0)*(wr0 + wr1);
				denom += kernelS(v, 0)*(wr0*p0 + wr1 * p1);
			}
			for (int v = 1; v <= r; ++v)
				for (int u = 1; u <= r; ++u)
				{
					double p00 = src(atN(y - v), atW(x - u));
					double p01 = src(atS(y + v), atW(x - u));
					double p10 = src(atN(y - v), atE(x + u));
					double p11 = src(atS(y + v), atE(x + u));
					int t00 = int(p00);
					int t01 = int(p01);
					int t10 = int(p10);
					int t11 = int(p11);
					double wr00 = kernelR[abs(t00 - t)];
					double wr01 = kernelR[abs(t01 - t)];
					double wr10 = kernelR[abs(t10 - t)];
					double wr11 = kernelR[abs(t11 - t)];
					numer += kernelS(v, u)*(wr00 + wr01 + wr10 + wr11);
					denom += kernelS(v, u)*(wr00*p00 + wr01 * p01 + wr10 * p10 + wr11 * p11);
				}
			dst(y, x) = denom / numer;
		}
}
//const bool sw_imshow = true;
Mat test_bilateral_filter(cv::Mat imageS, double sigmaS, double sigmaR, double tol, bool sw_imshow, bool last)
{
	//cv::Mat imageS = cv::imread(pathS, -1);
	if (imageS.empty())
	{
		std::cerr << "Source image loading failed!" << std::endl;
		return imageS;
	}

	std::cerr << "[Loaded Image]" << std::endl;
	std::cerr << cv::format("Source: \"\"  # (w,h,ch)=(%d,%d,%d)",  imageS.cols, imageS.rows, imageS.channels()) << std::endl;
	std::cerr << "[Filter Parameters]" << std::endl;
	std::cerr << cv::format("sigmaS=%f  sigmaR=%f  tolerance=%f", sigmaS, sigmaR, tol) << std::endl;

	cv::Mat src;
	imageS.convertTo(src, CV_64F);

	std::vector<cv::Mat_<double> > srcsp;
	cv::split(src, srcsp);

	std::vector<cv::Mat_<double> > dstsp0(src.channels());
	//std::vector<cv::Mat_<double> > dstsp1(src.channels());
	for (int c = 0; c<int(src.channels()); ++c)
	{
		dstsp0[c] = cv::Mat_<double>(src.size());
		//dstsp1[c] = cv::Mat_<double>(src.size());
	}

	cv::TickMeter tm;
	// Original bilateral filtering
	tm.start();
	//for (int i = 0; i < 1; i++) {
		for (int c = 0; c < src.channels(); ++c)
			apply_bilateral_filter(srcsp[c], dstsp0[c], sigmaS, sigmaR);
	//}
	
		if (last)
		{
			for (int c = 0; c < src.channels(); ++c)
			{
				for (int i = 0; i < src.rows; i++)
				{
					for (int j = 0; j < src.cols; j++)
					{
						double p = dstsp0[c](i, j);
						p = ((p /255.0f) / 24) * 24;
						dstsp0[c](i, j) = p;
					}
				}
			}
		}
	tm.stop();
	std::cerr << cv::format("Original BF:     %7.1f [ms]", tm.getTimeMilli()) << std::endl;
	tm.reset();

	cv::Mat dst0, dst1;
	cv::merge(dstsp0, dst0);

	const double tone = 256.0;
	if (sw_imshow)
	{
		cv::imshow("dst0", dst0 / (tone - 1.0));/// (tone - 1.0));
		cv::waitKey();
	}
	return dst0;
}

Mat run(cv::Mat imageS, int nr) {
	const double tone = 256.0;

	const bool sw_imshow = false;
	const bool sw_imwrite = false;
	
	const double sigmaS = 2.0;
	const double sigmaR = 0.1*(tone - 1.0);
	const double tol = 0.1; // for compressive BF
	//cv::Mat imageS = cv::imread("Images/lenna1.png", -1);
	cv::Mat imageD, imageE, imageF, imageG;
	imageD = imageS.clone();
	for (int iteration = 0; iteration < nr; iteration++)
	{
		if (iteration == (nr - 1)) {
			imageD = test_bilateral_filter(imageD, sigmaS, sigmaR, tol, false, true);
		}
		else {
			imageD = test_bilateral_filter(imageD, sigmaS, sigmaR, tol, false, false);
		}
	}
	//imageD = test_bilateral_filter(imageS, sigmaS, sigmaR, tol,false,false);
	//imageE = test_bilateral_filter(imageD, sigmaS, sigmaR, tol, false, false);
	//imageF = test_bilateral_filter(imageE, sigmaS, sigmaR, tol, false, false);
	//imageG = test_bilateral_filter(imageF, sigmaS, sigmaR, tol, false, true);
	if (sw_imshow)
	{
		cv::imshow("imageD", imageD);
		cv::waitKey();
	}
	return imageD;

}

Mat colorToGray2(Mat imagine) {
	//Mat imagine = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_COLOR);
	Mat imagineFinala = Mat(imagine.rows, imagine.cols, CV_8UC1);

	for (int i = 0; i < imagine.rows; i++)
		for (int j = 0; j < imagine.cols; j++) {
			Vec3b BGR = imagine.at<Vec3b>(i, j);
			float blue = BGR[0];
			float green = BGR[1];
			float red = BGR[2];
			imagineFinala.at<uchar>(i, j) = (blue + green + red) / 3;
		}

	return imagineFinala;
}
Mat filtru_median(Mat src, int k) {
	int w = 2 * k + 1;

	int d = w / 2;
	Mat imgFiltruMedian = src.clone();
	vector<uchar> vec;

	for (int i = k; i < src.rows - k; i++) {
		for (int j = k; j < src.cols - k; j++) {

			vec.clear();

			for (int ll = -d; ll <= d; ll++) {
				for (int lc = -d; lc <= d; lc++) {
					vec.push_back(src.at<uchar>(i + ll, j + lc));
				}
			}

			sort(vec.begin(), vec.end());

			imgFiltruMedian.at<uchar>(i, j) = vec[w*w / 2];
		}
	}
	return imgFiltruMedian;

}



int adaptive_histograma(Mat img)
{
	int his[256];
	float p = 0.1;
	int k = 0.4;

	for (int i = 0; i < 256; i++) {
		his[i] = 0;
	}
	for (int i = 0; i < img.rows - 1; i++) {
		for (int j = 0; j < img.cols - 1; j++) {
			uchar pixel = img.at<uchar>(i, j);
			his[pixel]++;

		}
	}


	float NrNonMuchie = (1 - p) * (img.rows * img.cols - his[0]);

	int suma = 0;
	for (int i = 1; i < 255; i++) {
		suma += his[i];
		if (suma > NrNonMuchie) {
			return i;
		}
	}
	return 255;


}


Mat canny(Mat img) {
	//sobel

	float sobelY[9] =
	{ 1.0, 2.0, 1.0,
		0.0, 0.0, 0.0,
		-1.0, -2.0, -1.0 };


	float sobelX[9] =
	{ -1.0, 0.0, 1.0,
		-2.0, 0.0, 2.0,
		-1.0, 0.0, 1.0 };

	Mat kernelx = Mat(3, 3, CV_32FC1, sobelX);
	Mat kernely = Mat(3, 3, CV_32FC1, sobelY);


	//Mat img = imread("Images/lenna11.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	Mat tetaSobel = Mat(img.rows, img.cols, CV_8UC1);


	Mat dstx = conv(img, kernelx);
	Mat dsty = conv(img, kernely);

	int dir = 0;

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			float squareX = dstx.at<float>(i, j)* dstx.at<float>(i, j);
			float squareY = dsty.at<float>(i, j)* dsty.at<float>(i, j);

			dst.at<uchar>(i, j) = sqrt(squareX + squareY) / (4 * sqrt(2));

			float teta = atan2(dsty.at<float>(i, j), dstx.at<float>(i, j));

			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
			if ((teta >   PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
			if ((teta >  -PI / 8 && teta <   PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) dir = 2;
			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta <   -PI / 8)) dir = 3;
			tetaSobel.at<uchar>(i, j) = dir;

		}
	}


	imshow("original", img);
	imshow("dest", dst);
	imshow("Directie", tetaSobel);

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			switch (tetaSobel.at<uchar>(i, j)) {
			case 0:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 1:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j + 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j - 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 2:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i, j - 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i, j + 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 3:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j - 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j + 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			default:
				break;
			}
		}
	}


	imshow("Model dupa NMS", dst);

	int ph = adaptive_histograma(dst);// +25;
	int pl = 0.4 * ph;

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (dst.at<uchar>(i, j) < pl) {
				dst.at<uchar>(i, j) = 0;
			}
			if (dst.at<uchar>(i, j) > ph) {
				dst.at<uchar>(i, j) = STRONG;
			}
			if (dst.at<uchar>(i, j) > pl && dst.at<uchar>(i, j) < ph) {
				dst.at<uchar>(i, j) = WEAK;
			}
		}
	}

	printf("ph=%d\n", ph);
	printf("pl=%d\n", pl);
	Mat ddd = dst.clone();

	for (int i = 0; i < ddd.rows; i++) {
		for (int j = 0; j < ddd.cols; j++) {
			if (ddd.at<uchar>(i, j) == WEAK) {
				ddd.at<uchar>(i, j) = 0;
			}
		}
	}
	imshow("ddd ddd ddd", ddd);
	Mat modul = dst.clone();
	int dx[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	Mat visited = Mat::zeros(dst.size(), CV_8UC1);
	queue <Point> que;
	for (int i = 1; i < dst.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			if (dst.at<uchar>(i, j) == STRONG && visited.at<uchar>(i, j) == 0) {
				que.push(Point(j, i));
				while (!que.empty()) {
					Point oldest = que.front();
					int jj = oldest.x;
					int ii = oldest.y;
					que.pop();
					for (int n = 0; n < 8; n++) {
						if (dst.at<uchar>(ii + dx[n], jj + dy[n]) == WEAK && visited.at<uchar>(ii + dx[n], jj + dy[n]) == 0) {
							dst.at<uchar>(ii + dx[n], jj + dy[n]) = STRONG;
							visited.at<uchar>(ii + dx[n], jj + dy[n]) = 1;
							que.push(Point(jj + dy[n], ii + dx[n]));
						}
					}
					visited.at<uchar>(i, j) = 1;
				}
			}
		}
	}

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (dst.at<uchar>(i, j) == WEAK) {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	imshow(" final ", dst);
	return dst;
}

void proiect() {

	clock_t tStart = clock();
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src1 = imread(fname, CV_LOAD_IMAGE_COLOR);
		imshow("Sursa: ", src1);
		Mat src = colorToGray2(src1);
		imshow("colorToGray2: ", src);

		//Mat src = imread("Images/lenna1.png", CV_LOAD_IMAGE_GRAYSCALE);
		Mat imgFiltruMedian = filtru_median(src, 3);
		imgFiltruMedian = filtru_median(imgFiltruMedian, 3);
		imshow("imgFiltruMedian", imgFiltruMedian);
		//Mat dilatare = dil_img_test(imgFiltruMedian,1);
		//imshow("dilatare", dilatare);
		//double kk = 0.99;
		//int pH = 50;//50;
		//int pL = (int)kk*pH;
		Mat  imgCanny, gauss;
		//GaussianBlur(dilatare, gauss, Size(5, 5), 0.8, 0.8);
		//GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		//imshow("gauss", gauss);
		//Canny(gauss, imgCanny, pL, pH, 3);
		//imshow("imgCanny", imgCanny);
		imgCanny = canny(imgFiltruMedian);

		Mat negativImgCanny;
		negativImgCanny = negativul_imaginii2(imgCanny);
		imshow("negativImgCanny", negativImgCanny);

		//Mat imgCannyDilatat = dilatareImagin(imgCanny);
		//imshow("imgCannyDilatat", imgCannyDilatat);

		//Mat negativImgCannyDilatat;
		//negativImgCannyDilatat = negativul_imaginii2(imgCannyDilatat);
		//imshow("negativImgCannyDilatat", negativImgCannyDilatat);

	
		cv::Mat imageBilateralFilter;
		imageBilateralFilter = run(src1, 2);
		cv::Mat srcc;
		imageBilateralFilter.convertTo(srcc, CV_64F);

		std::vector<cv::Mat_<double> > srcsp;
		cv::split(srcc, srcsp);

		std::vector<cv::Mat_<double> > dstsp0(srcc.channels());
		for (int c = 0; c<int(srcc.channels()); ++c)
		{
			dstsp0[c] = cv::Mat_<double>(srcc.size());
		}
		for (int c = 0; c < int(srcc.channels()); c++)
		{
			for (int i = 0; i < srcc.rows; i++)
			{
				for (int j = 0; j < srcc.cols; j++)
				{
					if (negativImgCanny.at<uchar>(i, j) == 0)
					{
						dstsp0[c](i, j) = 0;
					}
					else
					{
						dstsp0[c](i, j) = srcsp[c](i, j);
					}
				}
			}
		}
		cv::Mat dst0,toShow;
		cv::merge(dstsp0, dst0);
		dst0.convertTo(toShow, CV_8UC3, 255.0);
		std::string resultName = "C://Users//DanB//Desktop//imgNr";
		std::string extension = ".jpg";
		srand(time(NULL));
		int nrrrr = rand() % 10001 + 1;
		resultName = resultName + std::to_string(nrrrr) + extension;
		imwrite(resultName, toShow);
		if (true)
		{
			cv::imshow("dst0", dst0);
			cv::waitKey();
		}

		printf("\nTime taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		

	}
		
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");

		printf(" 1 - proiect\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			proiect();
			break;
		}
	} while (op != 0);
	return 0;
}