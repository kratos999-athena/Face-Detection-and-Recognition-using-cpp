#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include<fstream>

using namespace cv;
using namespace std;

class facedetection {
private:
    std::string imagepath;
    Mat image;
    Mat grayscale;

public:
    facedetection(const std::string& str) : imagepath(str) {
        image = imread(imagepath); 
        if (image.empty()) {
            cout << "Error: Could not load image!" << endl;
            exit(1); 
        }
    }

    void convertToGrayscale() {
        cvtColor(image, grayscale, COLOR_BGR2GRAY);
        imshow("Grayscale Output", grayscale); 
    }

    std::vector<Rect> haarcascade() {
        cvtColor(image, grayscale, COLOR_BGR2GRAY); 
        CascadeClassifier face_cascade;

        if (!face_cascade.load("/haarcascade_frontalface_default.xml")) {
            cout << "Error loading Haar Cascade!" << endl;
            return {};
        }

        std::vector<Rect> faces;
        face_cascade.detectMultiScale(grayscale, faces, 1.1, 5, 0, Size(30, 30));

        return faces;
    }

    void showDetectedFaces(const std::vector<Rect>& faces) {
        Mat reimage;
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(image, faces[i], Scalar(255, 0, 0), 2); 

        }
        resize(image, reimage, Size(720,720));
        imshow("Face Detection", reimage); 
        waitKey(0); 
    }
};

int main() {
    std::string path = "";
    std::ofstream file{ "",std::ios::app };
    facedetection f{ path };
    std::vector<Rect> detectedFaces = f.haarcascade();
    Mat image = imread(path);
    //////////////CROPPING IMAGE
    for (size_t i = 0; i < detectedFaces.size(); i++) {
        cout << "Face " << i + 1 << ": x=" << detectedFaces[i].x
            << ", y=" << detectedFaces[i].y
            << ", width=" << detectedFaces[i].width
            << ", height=" << detectedFaces[i].height << endl;
    }
    Mat  grayscale;
    for (size_t i = 0; i < detectedFaces.size(); i++) {
        Mat crop = image(Rect(detectedFaces[i].x, detectedFaces[i].y, detectedFaces[i].width, detectedFaces[i].height));
        cvtColor(crop, grayscale, COLOR_BGR2GRAY);
        //////////////// done to remove the smaller portionscoming in haarcascade but be mindful in case face is smaller than 300X300 size is essential
        Size imgsize = grayscale.size();
        if (imgsize.width && imgsize.height < 330) {
            continue;
        }
        //////////// 20x20 pixel cropping
        int x = 0, y = 0;
        int histogram[256];
        for (int a = 0; a < 256; a++) {
            histogram[a] = 0;
        }
        double normalisedhistogram[256];
        Mat twentycrop,resized;
        resize(grayscale, resized, Size(720, 720));
        for (int i = 0; i < 720; i = i + 20) {
            for (int j = 0; j < 720; j = j + 20) {
                twentycrop = resized(Rect(0+i,0+j,20,20));
                int intensitymatrix[20][20];
                for (int a = 0; a < 20; a++) {
                    for (int b = 0; b < 20; b++) {
                        uchar pixel_value = twentycrop.at<uchar>(0 + a, 0 + b);
                        intensitymatrix[a][b] = static_cast<int>(pixel_value);
                    }
                }
                /*for (int a = 0; a < 20; a++) {
                    for (int b = 0; b < 20; b++) {
                        std::cout << intensitymatrix[a][b] << " ";
                    }
                    std:: cout << "\n";
                }
                std::cout << "\n" << "------------------------" << "\n";*/
                ////////////////// converting into lbp////// logical error when dealing with bottom section of 20x20 pixel we will be assigning them 0 but in reality they will be a part of some matrix and in that matrix they will have some value
                int lbpmatrix[20][20];
                for (int a = 0; a < 20; a++) {
                    for (int b = 0; b < 20; b++) {
                        int sum = 0;
                        if (a == 0 || a == 19 || b == 0 || b == 19) {
                            lbpmatrix[a][b] = 0;
                            continue;
                        }
                        else {
                            if (intensitymatrix[a - 1][b - 1] >= intensitymatrix[a][b]) {
                                sum += 128;
                            }
                            if (intensitymatrix[a - 1][b] >= intensitymatrix[a][b]) {
                                sum += 64;
                            }
                            if (intensitymatrix[a - 1][b + 1] >= intensitymatrix[a][b]) {
                                sum += 32;
                            }
                            if (intensitymatrix[a][b - 1] >= intensitymatrix[a][b]) {
                                sum += 16;
                            }
                            if (intensitymatrix[a][b + 1] >= intensitymatrix[a][b]) {
                                sum += 8;
                            }
                            if (intensitymatrix[a + 1][b - 1] >= intensitymatrix[a][b]) {
                                sum += 4;
                            }
                            if (intensitymatrix[a + 1][b] >= intensitymatrix[a][b]) {
                                sum += 2;
                            }
                            if (intensitymatrix[a + 1][b + 1] >= intensitymatrix[a][b]) {
                                sum += 1;
                            }
                        }
                        lbpmatrix[a][b] = sum;
                    }
                }
                for (int a = 0; a < 20; a++) {
                    for (int b = 0; b < 20; b++) {
                        histogram[lbpmatrix[a][b]] += 1;
                    }
                   
                }
                

            }
        }
        for (int a = 0; a < 256; a++) {
            normalisedhistogram[a] =static_cast<long double>( histogram[a] )/ 518400;
            std::cout << normalisedhistogram[a] << " ";
            file << normalisedhistogram[a] << " ";
        }
        file << "\n";
        std::cout << "im the best";
        imshow("cropped image", resized);
        waitKey(500);
    }
    file.close();

    return 0;
}
