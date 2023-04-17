#include <boost/program_options.hpp>
#include <iostream>
#include <iterator>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>

int main(int argc, char* argv[])
{
    // Pass in model info via CLI args
    // clang-format off
    boost::program_options::options_description opts_desc
    ("usage: mfn [Options] imgfile\n\nOptions");
    opts_desc.add_options()
        ("model-proto", boost::program_options::value<std::string>()->required(), "gender model description file")
        ("model-weights", boost::program_options::value<std::string>()->required(), "gender model weights file")
        ("imgfile", boost::program_options::value<std::string>()->required(), "image file to process");
    // clang-format on
    boost::program_options::positional_options_description args_desc;
    args_desc.add("imgfile", 1);
    boost::program_options::variables_map vm;
    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv).options(opts_desc).positional(args_desc).run(), vm);
        boost::program_options::notify(vm);
    }
    catch (boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << opts_desc << std::endl;
        return 1;
    }
    if (!vm.count("imgfile"))
    {
        std::cerr << "Error: an image file must be specified" << std::endl << std::endl << opts_desc << std::endl;
        return 1;
    }

    // Tuned model parameters
    const double     genderInputScale    = 1.0;
    const cv::Size   genderInputBlobSize = cv::Size(227, 227);
    const cv::Scalar genderInputBlobMean = cv::Scalar(78.4263377603, 87.7689143744, 114.895847746);

    // Instantiate model
    cv::dnn::Net genderClassificationNet =
        cv::dnn::readNet(vm["model-weights"].as<std::string>(), vm["model-proto"].as<std::string>());
    genderClassificationNet.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

    // Read in image file
    cv::Mat image = cv::imread(vm["imgfile"].as<std::string>());
    if (image.empty())
    {
        std::cerr << "Error: failed to load image file " << vm["imgfile"].as<std::string>() << std::endl;
        return 1;
    }

    // Apply model
    try
    {
        cv::Mat genderInputBlob =
            cv::dnn::blobFromImage(image, genderInputScale, genderInputBlobSize, genderInputBlobMean, false);

        genderClassificationNet.setInput(genderInputBlob);
        std::vector<float> genderPredictions = genderClassificationNet.forward();

        const std::vector<std::string> genderClasses = {"m", "f"};
        std::cout << genderClasses[std::distance(genderPredictions.begin(),
                                                 std::max_element(genderPredictions.begin(), genderPredictions.end()))]
                  << std::endl;
    }
    catch (...)
    {
        std::cout << "n" << std::endl;
    }

    return 0;
}
