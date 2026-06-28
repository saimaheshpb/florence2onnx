#include "shared_types.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

// Test suite class
class SharedTypesTest {
private:
    void test_bounding_box() {
        std::cout << "Testing BoundingBox..." << std::endl;
        
        // Test constructor with individual values
        florence2::BoundingBox<float> box1(0.0f, 1.0f, 2.0f, 3.0f);
        assert(box1.xmin == 0.0f && box1.ymin == 1.0f && box1.xmax == 2.0f && box1.ymax == 3.0f);
        std::cout << "Individual value constructor: OK" << std::endl;
        
        // Test vector constructor
        std::vector<float> values = {4.0f, 5.0f, 6.0f, 7.0f};
        florence2::BoundingBox<float> box2(values);
        assert(box2.xmin == 4.0f && box2.ymin == 5.0f && box2.xmax == 6.0f && box2.ymax == 7.0f);
        std::cout << "Vector constructor: OK" << std::endl;
    }

    void test_coordinates() {
        std::cout << "\nTesting Coordinates..." << std::endl;
        
        // Test constructor with individual values
        florence2::Coordinates<float> coord1(10.0f, 20.0f);
        assert(coord1.x == 10.0f && coord1.y == 20.0f);
        std::cout << "Individual value constructor: OK" << std::endl;
        
        // Test vector constructor
        std::vector<float> values = {30.0f, 40.0f};
        florence2::Coordinates<float> coord2(values);
        assert(coord2.x == 30.0f && coord2.y == 40.0f);
        std::cout << "Vector constructor: OK" << std::endl;
    }

    void test_labeled_bounding_boxes() {
        std::cout << "\nTesting LabeledBoundingBoxes..." << std::endl;
        
        florence2::LabeledBoundingBoxes labeled_box;
        labeled_box.label = "test_label";
        labeled_box.bboxes.push_back(florence2::BoundingBox<float>(0.0f, 1.0f, 2.0f, 3.0f));
        
        assert(labeled_box.label == "test_label");
        assert(labeled_box.bboxes.size() == 1);
        std::cout << "LabeledBoundingBoxes: OK" << std::endl;
    }

    void test_florence_results() {
        std::cout << "\nTesting FlorenceResults..." << std::endl;
        
        florence2::FlorenceResults results;
        results.pure_text = "sample text";
        
        // Add a labeled OCR box
        florence2::LabeledOCRBox ocr_box;
        ocr_box.text = "OCR text";
        ocr_box.quad_box.push_back(florence2::Coordinates<float>(1.0f, 1.0f));
        results.ocr_bbox.push_back(ocr_box);
        
        assert(results.pure_text == "sample text");
        assert(results.ocr_bbox.size() == 1);
        assert(results.ocr_bbox[0].text == "OCR text");
        std::cout << "FlorenceResults: OK" << std::endl;
    }

public:
    void run_all_tests() {
        std::cout << "Starting SharedTypes tests...\n" << std::endl;
        
        try {
            test_bounding_box();
            test_coordinates();
            test_labeled_bounding_boxes();
            test_florence_results();
            
            std::cout << "\nAll tests passed successfully!" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Test failed with exception: " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    try {
        SharedTypesTest test_suite;
        test_suite.run_all_tests();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Tests failed: " << e.what() << std::endl;
        return 1;
    }
}