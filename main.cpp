#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include< fstream>


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

        if (!face_cascade.load("C:/Users/anurag Agarwal/source/repos/opencvcourse/opencvcourse/images/haarcascade_frontalface_default.xml")) {
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
        resize(image, reimage, Size(720, 720));
        imshow("Face Detection", reimage); 
        waitKey(0); 
    }
};
/*class filehandling {
private:
    std::string path_to_dataset{};
public:
    filehandling(const std::string& pth) :
        path_to_dataset{ pth } {
        if (!fs::exists(path_to_dataset) || !fs::is_directory(path_to_dataset) {

            std::cerr << "ERROR IN DATASET PATH";
        }
    }
    for (const auto& entry : fs::directory_iterator(main_folder_path)) {
        if (fs::is_directory(entry.path())) {
            std::string person_name = entry.path().filename().string(); // Get the folder name (person's name)
            std::cout << "Person: " << person_name << std::endl;

            // Iterate over the files inside the person's folder
                for (const auto& file_entry : fs::directory_iterator(entry.path())) {
                    if (fs::is_regular_file(file_entry.path())) {
                        std::string image_path = file_entry.path().string(); // Get the file path
                        std::cout << "Image: " << image_path << std::endl;
                    }
                }
        }
    }

};*/
/*class imagemodification {
private:
    Mat img;
public:
    imagemodification(const Mat& str):
    img(str){}

};*/
class normalised_histogram {
private:
    Mat image;// it is coming in grayscale,amke sure of that
public:
    normalised_histogram(const Mat& modified_image):
    image(modified_image){}
    std::vector<long double> normalisation() {
        Mat crop = image;
        int histogram[256];
        for (int a = 0; a < 256; a++) {
            histogram[a] = 0;
        }
        std::vector<long double> normalisedhistogram;
        Mat twentycrop, resized;
        resize(crop, resized, Size(720, 720));
        for (int i = 0; i < 720; i = i + 20) {
            for (int j = 0; j < 720; j = j + 20) {
                twentycrop = resized(Rect(0 + i, 0 + j, 20, 20));
                int intensitymatrix[20][20];
                for (int a = 0; a < 20; a++) {
                    for (int b = 0; b < 20; b++) {
                        uchar pixel_value = twentycrop.at<uchar>(0 + a, 0 + b);
                        intensitymatrix[a][b] = static_cast<int>(pixel_value);
                    }
                }
                    /// logical error when dealing with bottom section of 20x20 pixel we will be assigning them 0 but in reality they will be a part of some matrix and in that matrix they will have some value
                int lbpmatrix[20][20];
                for (int a = 0; a < 20; a++) {
                    for (int b = 0; b < 20; b++) {
                        int sum = 0;
                        if (a == 0 || a == 19 || b == 0 || b == 19) {
                            if (a == 0) {
                                lbpmatrix[a][b] = intensitymatrix[a + 1][b];
                            }
                            else if (a == 19) {
                                lbpmatrix[a][b] = intensitymatrix[a - 1][b];
                            }
                            else if (b == 0) {
                                lbpmatrix[a][b] = intensitymatrix[a][b + 1];
                            }
                            else if (b == 19) {
                                lbpmatrix[a][b] = intensitymatrix[a][b - 1];
                            }
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
            normalisedhistogram.emplace_back(static_cast<long double>(histogram[a]) / 518400);
        }
        return normalisedhistogram;
       
    }

};
class facerecognition {
private:
    std::vector <long double> nh{};
    long double chi_squarevalue{};
    long double iteration{};
public:
    facerecognition(std::vector <long double> str,long double x=0, long double y=0):
    nh(str),chi_squarevalue(x),iteration(y){}
    void chi_square(std::vector<long double> data_vec) {
        long double chi = 0;
        for (size_t i = 0; i < nh.size(); i++) {
            if ((nh[i] + data_vec[i]) != 0) {
                chi += ((nh[i] - data_vec[i]) * (nh[i] - data_vec[i])) / (nh[i] + data_vec[i]);
            }
        }
        chi_squarevalue += chi;
        
    }
    int finally() {
        std::ifstream file("C:/Users/anurag Agarwal/source/repos/opencvcourse/opencvcourse/images/dataset/trilokpuri/histogram12.txt");
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream stream(line); 

            std::vector<long double> numbers;
            iteration = iteration + 1;

            long double number;
            while (stream >> number) {
                numbers.push_back(number); 
            }
            chi_square(numbers);
        }
        long double mean = chi_squarevalue / iteration;
        //std::cout << mean<<" **  ";
        if (mean > 0.02) {
            return 0;
        }
        else {
            return 1;
        }

    }
    

};
int main() {
    // individually create a dataset for each person and then use
    // used haarcascade then lbp then formed histogram normalised the histogram then applied chi square method (compared two normalised histogram)
    // currently two histogram sets one in datasets folder
    // compiler was older than c++17 so was not able to apply modern file handling technique
    // very controlled environment is requiredwe need to set the threshold value ourselves and not able to handle multiple faces
    std::string path = "";
    facedetection f{ path };
    std::vector<Rect> detectedFaces = f.haarcascade();
    std::vector<Rect> finalface;
    Mat image = imread(path);
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
        /*Size imgsize = grayscale.size();
        if (imgsize.width && imgsize.height < 300) {
            continue;
        }*/
        normalised_histogram f{ grayscale };
        std::vector<long double> gfg = f.normalisation();
        facerecognition yui{ gfg };
        int hgh = yui.finally();
        if (hgh == 1) {
            finalface.emplace_back(detectedFaces[i]);
        }

       // logical error when dealing with bottom section of 20x20 pixel we will be assigning them 0 but in reality they will be a part of some matrix and in that matrix they will have some value
        
        //std::ofstream out{ "",std::ios::app };
        //for (int a = 0; a < 256; a++) {
        //    normalisedhistogram[a] = static_cast<long double>(histogram[a]) / 518400;
        //    std::cout << normalisedhistogram[a] << " ";
        //    out << normalisedhistogram[a] << " ";
        //}
        //imshow("output", grayscale);
        //waitKey(5000);
    }
    if (finalface.size() != 0) {
        rectangle(image, finalface[0], Scalar(0, 255, 0), 3);
        putText(image, "try to read the folder name", Point(finalface[0].x, finalface[0].y), FONT_HERSHEY_DUPLEX, 4, Scalar(0, 255, 0), 2);
        namedWindow("output", WINDOW_NORMAL);
        imshow("output", image);
        waitKey(0);
    }

    return 0;
}
