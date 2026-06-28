#include "post_processing/post_processor.h"
#include <iostream>
#include <cassert>
#include <cmath>

namespace florence2
{
    namespace test
    {

        class Florence2HuggingFaceFormatTest
        {
        private:
            static constexpr float EPSILON = 0.001f; // For float comparisons

            void test_caption_formats()
            {
                std::cout << "\nTesting Caption formats..." << std::endl;

                auto processor = std::make_unique<Florence2PostProcessor>();

                // Test all caption types
                std::vector<std::pair<std::string, TaskType>> caption_tests = {
                    {"<s>A cat sitting on a windowsill</s>", TaskType::CAPTION},
                    {"<s>A gray cat sitting on a wooden windowsill looking outside</s>", TaskType::DETAILED_CAPTION},
                    {"<s>A gray striped cat is perched on a wooden windowsill, gazing intently through the window at birds outside</s>", TaskType::MORE_DETAILED_CAPTION}};

                for (const auto &[input, task] : caption_tests)
                {
                    auto result = processor->PostProcessGeneration(input, task, {1000, 1000});
                    assert(!result.pure_text.empty());
                    std::cout << "Task: " << static_cast<int>(task) << ", Output: " << result.pure_text << std::endl;
                }

                std::cout << "Caption format tests: OK" << std::endl;
            }

            void test_phrase_grounding_format()
            {
                std::cout << "\nTesting Phrase Grounding format..." << std::endl;

                auto processor = std::make_unique<Florence2PostProcessor>();

                // Example: "A green car parked in front of a yellow building."
                std::string input = "green car<loc_100><loc_200><loc_300><loc_400>"
                                    "yellow building<loc_500><loc_600><loc_700><loc_800>";

                auto result = processor->PostProcessGeneration(
                    input,
                    TaskType::CAPTION_TO_PHRASE_GROUNDING,
                    {1000, 1000});

                assert(!result.bounding_boxes.empty());
                assert(result.bounding_boxes.size() == 2);
                assert(result.bounding_boxes[0].label == "green car");
                assert(result.bounding_boxes[1].label == "yellow building");

                std::cout << "Found " << result.bounding_boxes.size() << " phrases:" << std::endl;
                for (const auto &box : result.bounding_boxes)
                {
                    std::cout << "Label: " << box.label << ", BBox: ["
                              << box.bboxes[0].xmin << ", "
                              << box.bboxes[0].ymin << ", "
                              << box.bboxes[0].xmax << ", "
                              << box.bboxes[0].ymax << "]" << std::endl;
                }

                std::cout << "Phrase Grounding format test: OK" << std::endl;
            }

            void test_object_detection_format()
            {
                std::cout << "\nTesting Object Detection format..." << std::endl;

                auto processor = std::make_unique<Florence2PostProcessor>();

                std::string input = "car<loc_100><loc_200><loc_300><loc_400>"
                                    "person<loc_500><loc_600><loc_700><loc_800>";

                auto result = processor->PostProcessGeneration(
                    input,
                    TaskType::OD,
                    {1000, 1000});

                assert(!result.bounding_boxes.empty());
                assert(result.bounding_boxes.size() == 2);

                std::cout << "Found " << result.bounding_boxes.size() << " objects:" << std::endl;
                for (const auto &box : result.bounding_boxes)
                {
                    std::cout << "Label: " << box.label << ", BBox: ["
                              << box.bboxes[0].xmin << ", "
                              << box.bboxes[0].ymin << ", "
                              << box.bboxes[0].xmax << ", "
                              << box.bboxes[0].ymax << "]" << std::endl;
                }

                std::cout << "Object Detection format test: OK" << std::endl;
            }

            void test_dense_region_caption_format()
            {
                std::cout << "\nTesting Dense Region Caption format..." << std::endl;

                auto processor = std::make_unique<Florence2PostProcessor>();

                std::string input = "a red car in the parking lot<loc_100><loc_200><loc_300><loc_400>"
                                    "people walking on the sidewalk<loc_500><loc_600><loc_700><loc_800>";

                auto result = processor->PostProcessGeneration(
                    input,
                    TaskType::DENSE_REGION_CAPTION,
                    {1000, 1000});

                assert(!result.bounding_boxes.empty());
                assert(result.bounding_boxes.size() == 2);

                std::cout << "Found " << result.bounding_boxes.size() << " regions:" << std::endl;
                for (const auto &box : result.bounding_boxes)
                {
                    std::cout << "Caption: " << box.label << ", BBox: ["
                              << box.bboxes[0].xmin << ", "
                              << box.bboxes[0].ymin << ", "
                              << box.bboxes[0].xmax << ", "
                              << box.bboxes[0].ymax << "]" << std::endl;
                }

                std::cout << "Dense Region Caption format test: OK" << std::endl;
            }

            // TODO Verify Region Proposal Outputs
            void test_region_proposal_format()
            {
                std::cout << "\nTesting Region Proposal format..." << std::endl;

                auto processor = std::make_unique<Florence2PostProcessor>();

                // Debug the input format first
                std::string input = "<box><loc_100><loc_200><loc_300><loc_400></box>"
                                    "<box><loc_500><loc_600><loc_700><loc_800></box>";

                std::cout << "Testing with input: " << input << std::endl;

                auto result = processor->PostProcessGeneration(
                    input,
                    TaskType::REGION_PROPOSAL,
                    {1000, 1000});

                // Debug output
                if (!result.bounding_boxes.empty())
                {
                    std::cout << "Found " << result.bounding_boxes.size() << " regions:" << std::endl;
                    for (const auto &box : result.bounding_boxes)
                    {
                        if (!box.bboxes.empty())
                        {
                            std::cout << "BBox: ["
                                      << box.bboxes[0].xmin << ", "
                                      << box.bboxes[0].ymin << ", "
                                      << box.bboxes[0].xmax << ", "
                                      << box.bboxes[0].ymax << "]" << std::endl;
                        }
                    }
                }
                else
                {
                    std::cout << "No regions found!" << std::endl;
                }

                assert(!result.bounding_boxes.empty());
                std::cout << "Region Proposal format test: Found regions" << std::endl;

                std::cout << "Region Proposal format test: OK" << std::endl;
            }

            void test_ocr_formats()
            {
                std::cout << "\nTesting OCR formats..." << std::endl;

                auto processor = std::make_unique<Florence2PostProcessor>();

                // Simple OCR
                std::string simple_input = "<s>Text in the image: Hello World</s>";
                auto simple_result = processor->PostProcessGeneration(
                    simple_input,
                    TaskType::OCR,
                    {1000, 1000});

                assert(!simple_result.pure_text.empty());
                std::cout << "Simple OCR output: " << simple_result.pure_text << std::endl;

                // OCR with Region
                std::string region_input = "Hello<loc_100><loc_200><loc_300><loc_400><loc_500><loc_600><loc_700><loc_800>"
                                           "World<loc_150><loc_250><loc_350><loc_450><loc_550><loc_650><loc_750><loc_850>";

                auto region_result = processor->PostProcessGeneration(
                    region_input,
                    TaskType::OCR_WITH_REGION,
                    {1000, 1000});

                assert(!region_result.ocr_bbox.empty());
                assert(region_result.ocr_bbox.size() == 2);

                std::cout << "Found " << region_result.ocr_bbox.size() << " text regions:" << std::endl;
                for (const auto &ocr : region_result.ocr_bbox)
                {
                    std::cout << "Text: " << ocr.text << ", Quad points:" << std::endl;
                    for (size_t i = 0; i < ocr.quad_box.size(); i++)
                    {
                        std::cout << "  Point " << i << ": ("
                                  << ocr.quad_box[i].x << ", "
                                  << ocr.quad_box[i].y << ")" << std::endl;
                    }
                }

                std::cout << "OCR format tests: OK" << std::endl;
            }

        public:
            void run_all_tests()
            {
                std::cout << "Starting Florence2 HuggingFace Format tests..." << std::endl;

                try
                {
                    // test_caption_formats();
                    // test_phrase_grounding_format();
                    // test_object_detection_format();
                    // test_dense_region_caption_format();
                    test_region_proposal_format();
                    // test_ocr_formats();

                    std::cout << "\nAll HuggingFace format tests passed successfully!" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error: " << e.what() << std::endl;
                    throw;
                }
            }
        };

    } // namespace test
} // namespace florence2

int main()
{
    try
    {
        florence2::test::Florence2HuggingFaceFormatTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}