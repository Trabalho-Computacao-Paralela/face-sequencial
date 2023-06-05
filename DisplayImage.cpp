#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
using namespace std;
using namespace cv;
void detectAndDisplay( Mat frame );
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
                             "{help h||}"
                             "{face_cascade|/home/digitro/test/data/haarcascade/haarcascade_frontalface_alt.xml|Path to face cascade.}"
                             "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
                             "{camera|0|Camera device number.}");
    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
                  "You can use Haar or LBP features.\n\n" );
    parser.printMessage();
    String face_cascade_name = samples::findFile( parser.get<String>("face_cascade") );
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open( camera_device );
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}
void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4 );
        Mat faceROI = frame_gray( faces[i] );
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale( faceROI, eyes );
        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4 );
        }
    }
    //-- Show what you got
    imshow( "Capture - Face detection", frame );
}

// #include <dlib/image_processing/frontal_face_detector.h>
// #include <dlib/gui_widgets.h>
// #include <dlib/image_io.h>
// #include <iostream>

// using namespace dlib;
// using namespace std;

// // ----------------------------------------------------------------------------------------

// int main(int argc, char** argv)
// {  
//     try
//     {
//         if (argc == 1)
//         {
//             cout << "Give some image files as arguments to this program." << endl;
//             return 0;
//         }

//         frontal_face_detector detector = get_frontal_face_detector();
//         image_window win;

//         // Loop over all the images provided on the command line.
//         for (int i = 1; i < argc; ++i)
//         {
//             cout << "processing image " << argv[i] << endl;
//             array2d<unsigned char> img;
//             load_image(img, argv[i]);
//             // Make the image bigger by a factor of two.  This is useful since
//             // the face detector looks for faces that are about 80 by 80 pixels
//             // or larger.  Therefore, if you want to find faces that are smaller
//             // than that then you need to upsample the image as we do here by
//             // calling pyramid_up().  So this will allow it to detect faces that
//             // are at least 40 by 40 pixels in size.  We could call pyramid_up()
//             // again to find even smaller faces, but note that every time we
//             // upsample the image we make the detector run slower since it must
//             // process a larger image.
//             pyramid_up(img);

//             // Now tell the face detector to give us a list of bounding boxes
//             // around all the faces it can find in the image.
//             std::vector<rectangle> dets = detector(img);

//             cout << "Number of faces detected: " << dets.size() << endl;
//             // Now we show the image on the screen and the face detections as
//             // red overlay boxes.
//             win.clear_overlay();
//             win.set_image(img);
//             win.add_overlay(dets, rgb_pixel(255,0,0));

//             cout << "Hit enter to process the next image..." << endl;
//             cin.get();
//         }
//     }
//     catch (exception& e)
//     {
//         cout << "\nexception thrown!" << endl;
//         cout << e.what() << endl;
//     }
// }


// #include <opencv2/opencv.hpp>
// #include <opencv2/face.hpp>
// #include <iostream>

// using namespace std;
// using namespace cv;
// using namespace cv::face;

// int main() {
//     Mat imagem = imread("/mnt/c/Users/vinic/Pictures/20220716_180051.jpg");

//     CascadeClassifier faceDetector("/home/digitro/test/data/haarcascade/haarcascade_frontalface_alt.xml");

//     vector<Rect> faces;
//     faceDetector.detectMultiScale(imagem, faces);

//     Ptr<Facemark> facemark = FacemarkLBF::create();
//     facemark->loadModel("/mnt/c/Users/vinic/Downloads/lbfmodel.yaml");

//     for (int i = 0; i < faces.size(); i++) {
//         vector<Point2f> landmarks;
//         facemark->fit(imagem, faces[i], landmarks);

//         for (int j = 0; j < landmarks.size(); j++) {
//             circle(imagem, landmarks[j], 2, Scalar(0, 0, 255), -1);
//         }

//         rectangle(imagem, faces[i], Scalar(0, 255, 0), 2);
//     }

//     imshow("Imagem Original", imagem);
//     waitKey(0);

//     return 0;
// }
