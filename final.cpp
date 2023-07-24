#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include <open3d/Open3D.h>
#include <cmath>
#include <sstream>
#include <string>
#include <limits>
#include <chrono>
#include <iomanip>


struct Point3D
{
    float x;
    float y;
    float z;
};

struct DataPoint {
    std::string time;
    double latitude;
    double longitude;
};

std::string convertDurationToString(const std::chrono::seconds& duration) {
    int hours = duration.count() / 3600;
    int minutes = (duration.count() % 3600) / 60;
    int seconds = duration.count() % 60;

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds;

    return oss.str();
}

std::chrono::seconds convertTimeStringToDuration(const std::string& timeString) {
    int hours = 0, minutes = 0, seconds = 0;
    size_t hPos = timeString.find('h');
    size_t mPos = timeString.find('m');
    size_t sPos = timeString.find('s');

    if (hPos != std::string::npos)
        hours = std::stoi(timeString.substr(0, hPos));
    if (mPos != std::string::npos)
        minutes = std::stoi(timeString.substr(hPos + 1, mPos - hPos - 1));
    if (sPos != std::string::npos)
        seconds = std::stoi(timeString.substr(mPos + 1, sPos - mPos - 1));

    return std::chrono::hours(hours) + std::chrono::minutes(minutes) + std::chrono::seconds(seconds);
}

void writeToTextFile(const std::string& input1, const std::string& input2, const std::string& input3) {
    std::string filename = input2 + "_" + input3 + ".txt";

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error creating file: " << filename << std::endl;
        return;
    }

    file << "{file: " << input1 << ", lat: " << input2 << ", long: " << input3 << "}";

    file.close();

    std::cout << "File created: " << filename << std::endl;
}

double getTimeDifference(const std::string& time1, const std::string& time2) {
    // Convert time strings to seconds
    int hours1, minutes1, seconds1;
    sscanf(time1.c_str(), "%d:%d:%d", &hours1, &minutes1, &seconds1);
    int totalSeconds1 = hours1 * 3600 + minutes1 * 60 + seconds1;

    int hours2, minutes2, seconds2;
    sscanf(time2.c_str(), "%d:%d:%d", &hours2, &minutes2, &seconds2);
    int totalSeconds2 = hours2 * 3600 + minutes2 * 60 + seconds2;

    // Calculate time difference in seconds
    return std::abs(totalSeconds1 - totalSeconds2);
}

DataPoint findClosestTime(const std::string& filename, const std::string& timeValue) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return DataPoint();
    }

    std::string line;
    std::vector<DataPoint> dataPoints;

    // Read data from file and store in vector
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string time, coordinates;
        iss >> time >> coordinates;

        double latitude, longitude;
        sscanf(coordinates.c_str(), "(%lf,%lf)", &latitude, &longitude);

        dataPoints.push_back({time, latitude, longitude});
    }

    file.close();

    // Find closest time
    double minTimeDifference = std::numeric_limits<double>::max();
    DataPoint closestDataPoint;

    for (const auto& dataPoint : dataPoints) {
        double timeDifference = getTimeDifference(timeValue, dataPoint.time);
        if (timeDifference < minTimeDifference) {
            minTimeDifference = timeDifference;
            closestDataPoint = dataPoint;
        }
    }
    std::cout<< minTimeDifference << std::endl;

    return closestDataPoint;
}

DataPoint dpoint(const std::string& filename, const std::string& targetTime) {
    DataPoint d;

    DataPoint closestDataPoint = findClosestTime(filename, targetTime);

    if (closestDataPoint.time.empty()) {
        std::cout << "No data found or error reading file." << std::endl;
    } else {
        size_t startPos = closestDataPoint.time.find("(");
        size_t endPos = closestDataPoint.time.find(")");
        
        // Extract the substring between the parentheses
        std::string content = closestDataPoint.time.substr(startPos + 1, endPos - startPos - 1);
        
        std::cout << content << std::endl;

        std::vector<std::string> values;
        std::istringstream iss(content);
        std::string token;

        // Split the string based on the comma separator
        while (std::getline(iss, token, ',')) {
            values.push_back(token);
        }

        if (values.size() >= 2) {
            // Extract the latitude and longitude values
            d.latitude = std::stod(values[0]);
            d.longitude = std::stod(values[1]);
        }
    }
    d.time = closestDataPoint.time;

    return d;
}

bool parseDateTimeFromFileName(const std::string& fileName, std::string& date, std::string& time) {
    // Find the position of the second hyphen
    size_t secondHyphenPos = fileName.find("-", fileName.find("-") + 1);

    if (secondHyphenPos == std::string::npos) {
        return false; // Invalid file name format
    }

    // Extract the date and time substrings
    date = fileName.substr(secondHyphenPos + 1, 10);
    time = fileName.substr(secondHyphenPos + 12, 8);

    return true;
}

void adjustVideoBasedOnFirstFrame(const std::string& inputVideoPath, const std::string& outputVideoPath,int lower, int upper) {
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video file" << std::endl;
        return;
    }

    // Get the video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Define the lower and upper bounds for red in HSV color space//what does this do
    cv::Scalar lowerRed(0, lower, lower);
    cv::Scalar upperRed(10, upper, upper);
    
    // Read the first frame
    cv::Mat firstFrame;
    cap >> firstFrame;
    if (firstFrame.empty()) {
        std::cerr << "Failed to read first frame" << std::endl;
        return;
    }
    
    

    // Convert the first frame to HSV color space
    cv::Mat hsvFirstFrame;
    cv::cvtColor(firstFrame, hsvFirstFrame, cv::COLOR_BGR2HSV);

    // Threshold the first frame to get only red regions
    cv::Mat maskFirstFrame;
    cv::inRange(hsvFirstFrame, lowerRed, upperRed, maskFirstFrame);

    // Find contours of the red regions in the first frame
    std::vector<std::vector<cv::Point>> contoursFirstFrame;
    cv::findContours(maskFirstFrame, contoursFirstFrame, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the contour with the largest area in the first frame
    std::vector<cv::Point> maxContourFirstFrame;
    if (!contoursFirstFrame.empty()) {
        maxContourFirstFrame = *std::max_element(contoursFirstFrame.begin(), contoursFirstFrame.end(), [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
            return cv::contourArea(c1) < cv::contourArea(c2);
        });
    } else {
        std::cerr << "No laser line detected in the first frame" << std::endl;
        return;
    }

    // Fit a line to the contour in the first frame
    cv::Vec4f lineParamsFirstFrame;
    cv::fitLine(maxContourFirstFrame, lineParamsFirstFrame, cv::DIST_L2, 0, 0.01, 0.01);
    float slopeFirstFrame = lineParamsFirstFrame[1] / lineParamsFirstFrame[0];
    float angleFirstFrame = std::atan(slopeFirstFrame) * 180 / CV_PI;

    // Create a VideoWriter object to save the adjusted video
    cv::VideoWriter out(outputVideoPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(width, height));

    // Process the remaining frames in the video
    cv::Mat frame;
    while (cap.read(frame)) {

        
        float rotationAngle = angleFirstFrame;


        // Rotate the frame
        cv::Mat rotatedFrame;
        cv::Point2f frameCenter(static_cast<float>(frame.cols / 2), static_cast<float>(frame.rows / 2));
        cv::Mat rotationMatrix = cv::getRotationMatrix2D(frameCenter, rotationAngle, 1.0);
        cv::warpAffine(frame, rotatedFrame, rotationMatrix, frame.size());
        //cv::drawContours(rotatedFrame, std::vector<std::vector<cv::Point>>{maxContourFirstFrame}, 0, cv::Scalar(0, 255, 0), 2);


        // Write the adjusted frame to the output video
        out.write(rotatedFrame);
    }

    // Release the VideoCapture and VideoWriter objects
    cap.release();
    out.release();

    std::cout << "Video adjustment complete" << std::endl;
}

void cropAndCorrectFrames(const std::string& inputVideoPath, const std::string& outputVideoPath) {
    cv::Mat frame, croppedFrame, correctedFrame;
    cv::Mat perspectiveMatrix, invPerspectiveMatrix;
    
    cv::VideoCapture cap(inputVideoPath);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open the video file." << std::endl;
        return;
    }

    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    // Calculate the top and bottom boundaries for vertical cropping
    int top = 240;
    int bottom = 890;

    // Calculate the left and right boundaries for horizontal cropping
    int left = 340;
    int right = 1520;

    // Calculate the width and height of the cropped region
    int croppedWidth = right - left;
    int croppedHeight = bottom - top;
    
    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    writer.open(outputVideoPath, fourcc, fps, cv::Size(croppedWidth, croppedHeight), true);

    if (!writer.isOpened()) {
        std::cerr << "Failed to create the output video file." << std::endl;
        return;
    }

    while (cap.read(frame)) {
        // Crop the frame to the desired region
        croppedFrame = frame(cv::Range(top, bottom), cv::Range(left, right));

        // Define source and destination points for perspective transformation
        cv::Point2f srcPts[4] = { cv::Point2f(0, 0), cv::Point2f(croppedFrame.cols, 0),
                                  cv::Point2f(croppedFrame.cols, croppedFrame.rows), cv::Point2f(0, croppedFrame.rows) };
        cv::Point2f dstPts[4] = { cv::Point2f(0, 0), cv::Point2f(croppedFrame.cols, 0),
                                  cv::Point2f(croppedFrame.cols, croppedFrame.rows), cv::Point2f(0, croppedFrame.rows) };

        perspectiveMatrix = cv::getPerspectiveTransform(srcPts, dstPts);
        invPerspectiveMatrix = cv::getPerspectiveTransform(dstPts, srcPts);

        // Perform perspective transformation and correction
        cv::warpPerspective(croppedFrame, correctedFrame, perspectiveMatrix, cv::Size(croppedFrame.cols, croppedFrame.rows));

        // Write the corrected frame to the output video
        writer.write(correctedFrame);
    }
    cap.release();
    writer.release();
}

// Function to detect edges in a video
void detectEdgesNight(const std::string& inputVideoPath, const std::string& outputVideoPath, int upper, int lower) {
    cv::VideoCapture cap(inputVideoPath);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open the input video file." << std::endl;
        return;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer;
    int fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4');
    writer.open(outputVideoPath, fourcc, fps, cv::Size(width, height), false);

    if (!writer.isOpened()) {
        std::cerr << "Failed to create the output video file." << std::endl;
        return;
    }

    cv::Mat frame, edges;

    while (cap.read(frame)) {
        // Split the frame into BGR channels
        std::vector<cv::Mat> channels(3);
        cv::split(frame, channels);
        
        cv::Mat blurredFrame;
	cv::GaussianBlur(channels[2], blurredFrame, cv::Size(5, 5), 0);

        cv::Canny(blurredFrame, edges, lower, upper);

        
        // Resize edges to match the frame size
        cv::resize(edges, edges, frame.size());

        // Write the edges to the output video
        writer.write(edges);
    }

    cap.release();
    writer.release();
}



void topAndBottom(const std::string& inputVideoPath, const std::string& outputVideoPath, const std::string& searchDirection) {
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video file" << std::endl;
        return;
    }

    // Get the video properties
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // Create a VideoWriter object to save the output video
    cv::VideoWriter writer(outputVideoPath, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, cv::Size(width, height));

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat outputFrame = cv::Mat::zeros(frame.size(), CV_8UC3);  // Create an output frame with black background

        if (searchDirection == "bottom") {
            for (int col = 0; col < frame.cols; col++) {
                for (int row = frame.rows - 1; row >= 0; row--) {
                    cv::Vec3b pixel = frame.at<cv::Vec3b>(row, col);  // Get pixel value at (row, col)

                    // Check if the pixel is white
                    if (pixel[0] >= 200 && pixel[1] >= 200 && pixel[2] >= 200) {
                        outputFrame.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);  // Set the pixel to white in the output frame
                        break;  // Stop searching for white pixels in this column
                    }
                }
            }
        } else {  // Default search direction: top to bottom
            for (int col = 0; col < frame.cols; col++) {
                for (int row = 0; row < frame.rows; row++) {
                    cv::Vec3b pixel = frame.at<cv::Vec3b>(row, col);  // Get pixel value at (row, col)

                    // Check if the pixel is white
                    if (pixel[0] >= 200 && pixel[1] >= 200 && pixel[2] >= 200) {
                        outputFrame.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);  // Set the pixel to white in the output frame
                        break;  // Stop searching for white pixels in this column
                    }
                }
            }
        }

        writer.write(outputFrame);  // Write the output frame to the video
    }

    cap.release();  // Release the VideoCapture
    writer.release();  // Release the VideoWriter

    std::cout << "Video processing complete" << std::endl;
}

void processVideo(const std::string& inputVideoPath, const std::string& outputPLYPath, double filterHeight)
{
    cv::VideoCapture video(inputVideoPath);

    if (!video.isOpened())
    {
        std::cerr << "Failed to open the input video file." << std::endl;
        return;
    }

    std::ofstream plyFile(outputPLYPath);
    if (!plyFile.is_open())
    {
        std::cerr << "Failed to create the output PLY file." << std::endl;
        return;
    }

    cv::Mat frame;
    std::vector<Point3D> points;
    int frameNumber = 0;
    double adjuster;
    double pixelFilterHeight = -2*filterHeight;
    std::set<std::pair<float, float>> addedPoints;
    
    while(video.read(frame)){
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point> laserPixels;
        cv::findNonZero(gray, laserPixels);
	if( frameNumber < 3){
	    cv::Scalar average = cv::mean(laserPixels);
	    adjuster += average[1];
	}
	else{
	    adjuster = adjuster/3;
	    std::cout << "Adjuster: " << adjuster << std::endl;
	    break;
	}
	frameNumber++;
    }
    frameNumber = 0;
    video.set(cv::CAP_PROP_POS_FRAMES, 0);
    
    while (video.read(frame))
    {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point> laserPixels;
        cv::findNonZero(gray, laserPixels);
        
        
        for (const auto& pixel : laserPixels)
        {
            if(adjuster + pixelFilterHeight <= pixel.y){
                    float currentHeight = static_cast<float>(-(-adjuster + pixel.y)) /2.0;
		    if (addedPoints.find({ pixel.x, frameNumber }) == addedPoints.end())
		    {
			
				Point3D point;
				point.x = static_cast<float>(pixel.x)*(59.0/640.0);
				point.y = static_cast<float>(frameNumber)*(176/60.0);
				point.z = currentHeight ;
				points.push_back(point);
				addedPoints.insert({ pixel.x, frameNumber });
			
		    }
            }
        }

        frameNumber++;
    }

    plyFile << "ply\n";
    plyFile << "format ascii 1.0\n";
    plyFile << "element vertex " << points.size() << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "end_header\n";

    for (const auto& point : points)
    {
        plyFile << point.x << " " << point.y << " " << point.z << "\n";
    }

    plyFile.close();

    std::cout << "Point cloud saved to: " << outputPLYPath << std::endl;
}

void filterPointCloud(const std::string& inputPLYPath, const std::string& outputPLYPath)
{
    auto pcd = open3d::io::CreatePointCloudFromFile(inputPLYPath);

    auto [cl1, ind1] = pcd->RemoveStatisticalOutliers(200, 0.1);
    auto inlierCloud1 = pcd->SelectByIndex(ind1);

    auto [cl2, ind2] = inlierCloud1->RemoveStatisticalOutliers(100, 4.0);
    auto inlierCloud2 = inlierCloud1->SelectByIndex(ind2);

    open3d::io::WritePointCloud(outputPLYPath, *inlierCloud2);

    //open3d::visualization::DrawGeometries({inlierCloud2});
}

void offsetPointCloudY(const open3d::geometry::PointCloud& pointCloud, const std::string& time, const std::string& timeLat)
{
    auto pcd = std::make_shared<open3d::geometry::PointCloud>(pointCloud);

    // Get the first point
    const auto& firstPoint = pcd->points_.front();
    const auto& lastPoint = pcd->points_.back();
    
    
    double middle = (lastPoint[1]  - firstPoint[1] ) * 0.5;
    
    double offsetY = firstPoint[1]+middle;
    
    int frame = static_cast<int>(offsetY * (60.0 / 176.0));
    // save this to a text file.. eventually upload a text file with gps for each save frame
    
    std::chrono::seconds duration = convertTimeStringToDuration(time);
                 
	 std::chrono::seconds offset(frame / 60);
	 std::chrono::seconds newDuration = duration + offset;
	 
	 std::string timetoFind = convertDurationToString(newDuration);
	 
	 DataPoint result = dpoint(timeLat, timetoFind);//temporary as i sleep
	 std::string filename;
	 if (result.time.empty()) {
	    std::cout << "No data found or error reading file." << std::endl;
	    result.latitude = -100;
	    result.longitude = -100;
	    filename = "error.ply";
	} else {
	    // Cast latitude and longitude to strings
	    std::string latString = std::to_string(result.latitude);
	    std::string longString = std::to_string(result.longitude);

	    // Construct the filename
	    filename = latString + "_" + longString + ".ply";
	}
	 
    
    //std::cout<< frame <<std::endl;

    // Offset all y positions based on the first point
    for (auto& point : pcd->points_) {
        point[1] -= offsetY;
    }

    // Save the modified point cloud to a new file
    open3d::io::WritePointCloud(filename, *pcd);

    std::cout << "Offset point cloud saved as: " << filename << std::endl;
    
}



void segmentAndVisualizePointCloud(const std::string& inputFile, const std::string& time, const std::string& timelat)
{
    // Read the point cloud file
    auto pcd = open3d::io::CreatePointCloudFromFile(inputFile);

    if (pcd == nullptr) {
        std::cout << "Failed to read the point cloud file." << std::endl;
        return;
    }

    // Get the first and last points
    const auto& firstPoint = pcd->points_.front();
    const auto& lastPoint = pcd->points_.back();

    double minY = firstPoint[1];
    double maxY = lastPoint[1];

    // Calculate the segment size based on the range of y positions
    double segmentSize = (maxY - minY) / 10.0;

    // Create a vector to store each cluster's point cloud as shared_ptr
    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> clusters(10);

    // Segment the point cloud based on y position
    for (const auto& point : pcd->points_) {
        int clusterIndex = static_cast<int>((point[1] - minY) / segmentSize);
        if (clusterIndex >= 0 && clusterIndex < 10) {
            if (!clusters[clusterIndex]) {
                clusters[clusterIndex] = std::make_shared<open3d::geometry::PointCloud>();
            }
            clusters[clusterIndex]->points_.emplace_back(point);
        }
    }
    // Visualize the segmented point clouds one at a time
    for (int i = 0; i < 10; ++i) {
        if (clusters[i] && clusters[i]->points_.size() > 0) {
            open3d::visualization::DrawGeometries({clusters[i]});
            open3d::visualization::RenderOption().background_color_ = {0, 0, 0};

             std::string userInput;

    	     std::cout << "Save (y/n): ";
             std::getline(std::cin, userInput);
             
             if(userInput == "y" || userInput == "Y" ){
                 //std:string outputFile = "test"+std::to_string(i)+".ply"
                 offsetPointCloudY(*clusters[i],time, timelat);
             }
        }
    }
}


int main(int argc, char* argv[]) {

    bool runSecondCode = true;
    double filterHeight;
    
    std::string input_video_path0 = argv[1];
    std::string outputVideoPath0 = "adjusted.mkv";
    std::string outputVideoPath1 = "output_video.mkv";
    std::string outputVideoPath2 = "grey.mkv";
    std::string outputVideoPath3 = "merged.mkv";
    std::string outputPLYPath = "output_cloud.ply";
    std::string outputPLYPath1 = "removed.ply";
    std::string timeLat = argv[2];
    
    
    //adjustVideoBasedOnFirstFrame(input_video_path0, outputVideoPath0,150,255);
    cropAndCorrectFrames(input_video_path0, outputVideoPath1);
    
        
    int upper = 100;
    int lower = 50;
        
    detectEdgesNight(outputVideoPath1, outputVideoPath2,upper,lower);
    
    topAndBottom(outputVideoPath2, outputVideoPath3, "top");
    
    
    if(argv[3] == NULL){
        filterHeight = 2.0;
    }
    else filterHeight = std::stod(argv[3]);
    	
    
    
    processVideo(outputVideoPath3, outputPLYPath, filterHeight);
    
    
    filterPointCloud(outputPLYPath,outputPLYPath1);
    
    std::string date, time;
    
    parseDateTimeFromFileName(input_video_path0, date, time);
    
    segmentAndVisualizePointCloud(outputPLYPath1,time,timeLat);

    return 0;
}

