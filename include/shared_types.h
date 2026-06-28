// include/shared_types.h
#pragma once

#include <string>
#include <vector>
#include <type_traits>

namespace florence2 {

/*
 * C++ Port Changes from C#:
 * 1. Changed 'enum' to 'enum class' for type safety and scoping
 * 2. Removed 'public' modifier as C++ enums are public by default
 * 3. Changed naming from TaskTypes to TaskType following C++ convention
 */
enum class TaskType {
    OCR,
    OCR_WITH_REGION,
    CAPTION,
    DETAILED_CAPTION,
    MORE_DETAILED_CAPTION,
    OD,
    DENSE_REGION_CAPTION,
    CAPTION_TO_PHRASE_GROUNDING,
    REFERRING_EXPRESSION_SEGMENTATION, // TODO: not working properly, generated tokens don't seem to be ok
    REGION_TO_SEGMENTATION,
    OPEN_VOCABULARY_DETECTION,
    REGION_TO_CATEGORY,
    REGION_TO_DESCRIPTION,
    REGION_TO_OCR,
    REGION_PROPOSAL
};

/*
 * C++ Port Changes from C#:
 * 1. Replaced C# generic constraints with static_assert
 * 2. Changed properties to public members (C++ convention for data structures)
 * 3. Used member initializer lists in constructors
 * 4. Added explicit keyword to prevent implicit conversions
 * 5. Added value initialization for numeric types
 * 6. Changed array parameter to std::vector for safety and convenience
 */
template<typename T>
class BoundingBox {
public:
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    BoundingBox() = default;
    
    explicit BoundingBox(const std::vector<T>& values) {
        if (values.size() >= 4) {
            xmin = values[0];
            ymin = values[1];
            xmax = values[2];
            ymax = values[3];
        }
    }

    BoundingBox(T xmin_, T ymin_, T xmax_, T ymax_)
        : xmin(xmin_), ymin(ymin_), xmax(xmax_), ymax(ymax_) {}

    T xmin{};  // Value initialization added
    T ymin{};
    T xmax{};
    T ymax{};
};

/*
 * C++ Port Changes from C#:
 * 1. Similar changes as BoundingBox class
 * 2. Simplified type constraints using static_assert
 * 3. Changed array constructor to use std::vector
 */
template<typename T>
class Coordinates {
public:
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");

    Coordinates() = default;
    
    explicit Coordinates(const std::vector<T>& values) {
        if (values.size() >= 2) {
            x = values[0];
            y = values[1];
        }
    }

    Coordinates(T x_, T y_) : x(x_), y(y_) {}

    T x{};
    T y{};
};

/*
 * C++ Port Changes from C#:
 * 1. Changed array to std::vector for dynamic sizing
 * 2. Changed properties to public members
 * 3. Used snake_case naming convention for C++
 */
class LabeledBoundingBoxes {
public:
    std::vector<BoundingBox<float>> bboxes;
    std::string label;
};

/*
 * C++ Port Changes from C#:
 * 1. Changed Coordinates[] to std::vector
 * 2. Changed properties to public members
 * 3. Used snake_case naming convention
 */
class LabeledOCRBox {
public:
    std::vector<Coordinates<float>> quad_box;
    std::string text;
};

/*
 * C++ Port Changes from C#:
 * 1. Changed List<T> to std::vector
 * 2. Changed properties to public members
 * 3. Used snake_case naming convention
 */
class LabeledPolygon {
public:
    std::string label;
    std::vector<Coordinates<float>> polygon;
    std::vector<BoundingBox<float>> bboxes;
};

/*
 * C++ Port Changes from C#:
 * 1. Replaced C# List<T> with std::vector
 * 2. Changed properties to public members
 * 3. Used snake_case naming convention
 */
class FlorenceResults {
public:
    std::vector<LabeledOCRBox> ocr_bbox;
    std::string pure_text;
    std::vector<LabeledBoundingBoxes> bounding_boxes;
    std::vector<LabeledPolygon> polygons;
};



} // namespace florence2