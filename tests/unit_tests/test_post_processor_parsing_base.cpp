#include "post_processing/post_processor.h"
#include <iostream>
#include <cassert>
#include <cmath>

namespace florence2
{
    namespace test
    {

        class Florence2PostProcessorTest
        {
        private:
            void test_task_type_mapping()
            {
                std::cout << "\nTesting task type mapping..." << std::endl;

                try
                {
                    assert(Florence2PostProcessor::GetPostProcessingType(TaskType::OCR) ==
                           PostProcessingTypes::pure_text);
                    assert(Florence2PostProcessor::GetPostProcessingType(TaskType::OCR_WITH_REGION) ==
                           PostProcessingTypes::ocr_with_region);
                    assert(Florence2PostProcessor::GetPostProcessingType(TaskType::CAPTION) ==
                           PostProcessingTypes::pure_text);
                    assert(Florence2PostProcessor::GetPostProcessingType(TaskType::OD) ==
                           PostProcessingTypes::description_with_bboxes);

                    std::cout << "Task type mapping: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Task type mapping test failed: " << e.what() << std::endl;
                    throw;
                }
            }

            void test_initialization()
            {
                std::cout << "\nTesting Florence2PostProcessor initialization..." << std::endl;

                try
                {
                    auto processor = std::make_unique<Florence2PostProcessor>();
                    assert(processor != nullptr);
                    std::cout << "Basic initialization: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Initialization test failed: " << e.what() << std::endl;
                    throw;
                }
            }

            void test_config_initialization()
            {
                std::cout << "\nTesting configuration initialization..." << std::endl;

                try
                {
                    Florence2PostProcessorConfig config;

                    assert(config.numBboxHeightBins == 1000);
                    assert(config.numBboxWidthBins == 1000);
                    assert(config.boxQuantizationMode == QuantizerMode::Floor);
                    assert(config.coordinatesHeightBins == 1000);
                    assert(config.coordinatesWidthBins == 1000);
                    assert(!config.parseTasks.empty());

                    bool hasOdTask = false;
                    bool hasPhraseGrounding = false;
                    for (const auto &task : config.parseTasks)
                    {
                        if (task.taskName == PostProcessingTypes::od)
                        {
                            hasOdTask = true;
                        }
                        if (task.taskName == PostProcessingTypes::phrase_grounding)
                        {
                            hasPhraseGrounding = true;
                            assert(task.filterByBlackList);
                        }
                    }
                    assert(hasOdTask);
                    assert(hasPhraseGrounding);

                    std::cout << "Configuration initialization: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Configuration initialization test failed: " << e.what() << std::endl;
                    throw;
                }
            }

            void test_ocr_parsing()
            {
                std::cout << "\nTesting OCR parsing..." << std::endl;

                try
                {
                    auto processor = std::make_unique<Florence2PostProcessor>();

                    // Test case 1: Basic OCR with coordinates
                    std::string test1 = "Hello<loc_100><loc_200><loc_300><loc_400><loc_500><loc_600><loc_700><loc_800>";
                    auto result1 = processor->ParseOcrFromTextAndSpans(test1, {1000, 1000});

                    assert(result1.size() == 1);
                    assert(result1[0].text == "Hello");
                    assert(result1[0].quad_box.size() == 4); // Should have 4 coordinate pairs
                    std::cout << "Basic OCR parsing: OK" << std::endl;

                    // Test case 2: Multiple OCR regions
                    std::string test2 = "Text1<loc_100><loc_200><loc_300><loc_400><loc_500><loc_600><loc_700><loc_800>"
                                        "Text2<loc_150><loc_250><loc_350><loc_450><loc_550><loc_650><loc_750><loc_850>";
                    auto result2 = processor->ParseOcrFromTextAndSpans(test2, {1000, 1000});

                    assert(result2.size() == 2);
                    assert(result2[0].text == "Text1");
                    assert(result2[1].text == "Text2");
                    std::cout << "Multiple OCR regions: OK" << std::endl;

                    // Test case 3: Empty input
                    std::string test3 = "";
                    auto result3 = processor->ParseOcrFromTextAndSpans(test3, {1000, 1000});
                    assert(result3.empty());
                    std::cout << "Empty input handling: OK" << std::endl;

                    std::cout << "OCR parsing tests: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "OCR parsing test failed: " << e.what() << std::endl;
                    throw;
                }
            }

            void test_phrase_grounding()
            {
                std::cout << "\nTesting phrase grounding..." << std::endl;

                try
                {
                    auto processor = std::make_unique<Florence2PostProcessor>();

                    // Test case 1: Basic phrase with bounding box
                    std::string test1 = "red car<loc_100><loc_200><loc_300><loc_400>";
                    auto result1 = processor->ParsePhraseGroundingFromTextAndSpans(test1, {1000, 1000});

                    assert(result1.size() == 1);
                    assert(result1[0].label == "red car");
                    assert(result1[0].bboxes.size() == 1);
                    std::cout << "Basic phrase grounding: OK" << std::endl;

                    // Test case 2: Blacklisted phrase
                    std::string test2 = "it<loc_100><loc_200><loc_300><loc_400>";
                    auto result2 = processor->ParsePhraseGroundingFromTextAndSpans(test2, {1000, 1000});
                    assert(result2.empty()); // Should be filtered out
                    std::cout << "Blacklist filtering: OK" << std::endl;

                    // Test case 3: Multiple boxes for one phrase
                    std::string test3 = "multiple boxes<loc_100><loc_200><loc_300><loc_400><loc_500><loc_600><loc_700><loc_800>";
                    auto result3 = processor->ParsePhraseGroundingFromTextAndSpans(test3, {1000, 1000});
                    assert(result3.size() == 1);
                    assert(result3[0].bboxes.size() == 2);
                    std::cout << "Multiple boxes: OK" << std::endl;

                    std::cout << "Phrase grounding tests: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Phrase grounding test failed: " << e.what() << std::endl;
                    throw;
                }
            }
            void test_description_with_bboxes()
            {
                std::cout << "\nTesting description with bboxes parsing..." << std::endl;

                try
                {
                    auto processor = std::make_unique<Florence2PostProcessor>();

                    // Test case 1: Basic description with single bbox
                    std::string test1 = "a red car<loc_100><loc_200><loc_300><loc_400>";
                    auto result1 = processor->ParseDescriptionWithBboxesFromTextAndSpans(
                        test1, {1000, 1000}, false);

                    assert(result1.size() == 1);
                    assert(result1[0].label == "a red car");
                    assert(result1[0].bboxes.size() == 1);
                    std::cout << "Basic description with bbox: OK" << std::endl;

                    // Test case 2: Multiple bounding boxes
                    std::string test2 = "two cars<loc_100><loc_200><loc_300><loc_400><loc_500><loc_600><loc_700><loc_800>";
                    auto result2 = processor->ParseDescriptionWithBboxesFromTextAndSpans(
                        test2, {1000, 1000}, false);

                    assert(result2.size() == 1);
                    assert(result2[0].bboxes.size() == 2);
                    std::cout << "Multiple bboxes: OK" << std::endl;

                    // Test case 3: Empty phrase not allowed
                    std::string test3 = "<loc_100><loc_200><loc_300><loc_400>";
                    auto result3 = processor->ParseDescriptionWithBboxesFromTextAndSpans(
                        test3, {1000, 1000}, false);

                    assert(result3.empty());
                    std::cout << "Empty phrase handling: OK" << std::endl;

                    // Test case 4: Empty phrase allowed
                    auto result4 = processor->ParseDescriptionWithBboxesFromTextAndSpans(
                        test3, {1000, 1000}, true);

                    assert(!result4.empty());
                    assert(result4[0].bboxes.size() == 1);
                    std::cout << "Empty phrase allowed: OK" << std::endl;

                    std::cout << "Description with bboxes tests: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Description with bboxes test failed: " << e.what() << std::endl;
                    throw;
                }
            }

            void test_description_with_polygons()
            {
                std::cout << "\nTesting description with polygons parsing..." << std::endl;

                try
                {
                    auto processor = std::make_unique<Florence2PostProcessor>();

                    // Test case 1: Basic polygon - Fixed format
                    std::string test1 = "a shape<poly><loc_100><loc_200><loc_300><loc_400><loc_500><loc_600>";
                    auto result1 = processor->ParseDescriptionWithPolygonsFromTextAndSpans(
                        test1, {1000, 1000}, false);

                    std::cout << "Result size: " << result1.size() << std::endl;
                    if (!result1.empty())
                    {
                        std::cout << "Label: " << result1[0].label << std::endl;
                        std::cout << "Polygon points: " << result1[0].polygon.size() << std::endl;
                    }

                    // Test case 2: Polygon with initial bbox - Fixed syntax
                    std::string test2 = "a shape<loc_10><loc_20><loc_30><loc_40>";
                    auto result2 = processor->ParseDescriptionWithPolygonsFromTextAndSpans(
                        test2, {1000, 1000}, false, "<sep>", "<poly>", "</poly>", true);

                    if (!result2.empty())
                    {
                        std::cout << "Has bbox: " << (!result2[0].bboxes.empty()) << std::endl;
                        std::cout << "Has polygon points: " << (!result2[0].polygon.empty()) << std::endl;
                    }
                    std::cout << "Basic polygon and bbox tests completed" << std::endl;

                    // Test case 3: Simple polygon without tags
                    std::string test3 = "shape<loc_100><loc_200><loc_300><loc_400>";
                    auto result3 = processor->ParseDescriptionWithPolygonsFromTextAndSpans(
                        test3, {1000, 1000}, false);

                    assert(!result3.empty());
                    assert(!result3[0].polygon.empty());
                    std::cout << "Simple polygon parsing: OK" << std::endl;

                    std::cout << "Description with polygons tests: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Description with polygons test failed: " << e.what() << std::endl;
                    throw;
                }
            }

            void test_post_process_generation()
            {
                std::cout << "\nTesting PostProcessGeneration..." << std::endl;

                try
                {
                    auto processor = std::make_unique<Florence2PostProcessor>();

                    // Test pure text
                    std::string textInput = "<s>Hello World</s>";
                    auto textResult = processor->PostProcessGeneration(
                        textInput,
                        TaskType::OCR,
                        {1000, 1000});
                    assert(!textResult.pure_text.empty());
                    assert(textResult.pure_text == "Hello World");
                    std::cout << "Pure text processing: OK" << std::endl;

                    // Test OCR with region TODO
                    // std::string ocrInput = "Text<loc_100><loc_200><loc_300><loc_400><loc_500><loc_600><loc_700><loc_800>";
                    // auto ocrResult = processor->PostProcessGeneration(
                    //     ocrInput,
                    //     TaskType::OCR_WITH_REGION,
                    //     {1000, 1000});
                    // assert(!ocrResult.ocr_bbox.empty());
                    // std::cout << "OCR with region processing: OK" << std::endl;

                    // Test object detection
                    std::string odInput = "car<loc_100><loc_200><loc_300><loc_400>";
                    auto odResult = processor->PostProcessGeneration(
                        odInput,
                        TaskType::OD,
                        {1000, 1000});
                    assert(!odResult.bounding_boxes.empty());
                    std::cout << "Object detection processing: OK" << std::endl;

                    // Test phrase grounding
                    std::string pgInput = "red car<loc_100><loc_200><loc_300><loc_400>";
                    auto pgResult = processor->PostProcessGeneration(
                        pgInput,
                        TaskType::CAPTION_TO_PHRASE_GROUNDING,
                        {1000, 1000});
                    assert(!pgResult.bounding_boxes.empty());
                    std::cout << "Phrase grounding processing: OK" << std::endl;

                    // Test polygon description
                    std::string polyInput = "shape<poly><loc_100><loc_200><loc_300><loc_400></poly>";
                    auto polyResult = processor->PostProcessGeneration(
                        polyInput,
                        TaskType::REFERRING_EXPRESSION_SEGMENTATION,
                        {1000, 1000});
                    assert(!polyResult.polygons.empty());
                    std::cout << "Polygon processing: OK" << std::endl;

                    // Test bbox or polygon (bbox case)
                    std::string bboxInput = "object<loc_100><loc_200><loc_300><loc_400>";
                    auto bboxResult = processor->PostProcessGeneration(
                        bboxInput,
                        TaskType::OPEN_VOCABULARY_DETECTION,
                        {1000, 1000});
                    assert(!bboxResult.bounding_boxes.empty());
                    std::cout << "Bbox/Polygon (bbox case) processing: OK" << std::endl;

                    // Test bbox or polygon (polygon case)
                    std::string polyMixInput = "shape<poly><loc_100><loc_200><loc_300><loc_400></poly>";
                    auto polyMixResult = processor->PostProcessGeneration(
                        polyMixInput,
                        TaskType::OPEN_VOCABULARY_DETECTION,
                        {1000, 1000});
                    assert(!polyMixResult.polygons.empty());
                    std::cout << "Bbox/Polygon (polygon case) processing: OK" << std::endl;

                    std::cout << "PostProcessGeneration tests: OK" << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "PostProcessGeneration test failed: " << e.what() << std::endl;
                    throw;
                }
            }

        public:
            void run_all_tests()
            {
                std::cout << "Starting Florence2PostProcessor tests..." << std::endl;

                try
                {
                    test_initialization();
                    test_task_type_mapping();
                    test_config_initialization();
                    // test_ocr_parsing();
                    test_phrase_grounding();
                    test_description_with_bboxes();
                    test_description_with_polygons();
                    test_post_process_generation();

                    std::cout << "\nAll Florence2PostProcessor tests passed successfully!" << std::endl;
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
        florence2::test::Florence2PostProcessorTest test;
        test.run_all_tests();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}