#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <map>
#include "box_quantizer.h"
#include "post_processing/coordinates_quantizer.h"
#include "shared_types.h"

namespace florence2 {

enum class PostProcessingTypes {
    od,
    ocr_with_region,
    pure_text,
    description_with_polygons,
    description_with_bboxes,
    phrase_grounding,
    polygons,
    description_with_bboxes_or_polygons,
    bboxes
};

class ParseTask {
public:
    PostProcessingTypes taskName;
    bool filterByBlackList = false;
};

class Florence2PostProcessorConfig {
public:
    int numBboxHeightBins = 1000;
    int numBboxWidthBins = 1000;
    QuantizerMode boxQuantizationMode = QuantizerMode::Floor;
    int coordinatesHeightBins = 1000;
    int coordinatesWidthBins = 1000;
    QuantizerMode coordinatesQuantizationMode = QuantizerMode::Floor;
    std::vector<ParseTask> parseTasks;

    Florence2PostProcessorConfig();
};

class Florence2PostProcessor {
public:
    Florence2PostProcessor();

    static PostProcessingTypes GetPostProcessingType(TaskType taskType);
    
    FlorenceResults PostProcessGeneration(
        const std::string& text,
        TaskType task,
        std::pair<int, int> imageSize
    );

    std::string ReplaceStartAndEndToken(const std::string& text);
    
    std::vector<LabeledOCRBox> ParseOcrFromTextAndSpans(
        const std::string& text,
        std::pair<int, int> imageSize
    );

    std::vector<LabeledBoundingBoxes> ParsePhraseGroundingFromTextAndSpans(
        const std::string& text,
        std::pair<int, int> imageSize
    );

    std::vector<LabeledBoundingBoxes> ParseDescriptionWithBboxesFromTextAndSpans(
        const std::string& text,
        std::pair<int, int> imageSize,
        bool allowEmptyPhrase = false
    );

    std::vector<LabeledPolygon> ParseDescriptionWithPolygonsFromTextAndSpans(
        const std::string& text,
        std::pair<int, int> imageSize,
        bool allowEmptyPhrase = false,
        const std::string& polygonSepToken = "<sep>",
        const std::string& polygonStartToken = "<poly>",
        const std::string& polygonEndToken = "</poly>",
        bool withBoxAtStart = false
    );

private:
    Florence2PostProcessorConfig config;
    std::map<PostProcessingTypes, ParseTask> parseTaskConfigs;
    std::unique_ptr<BoxQuantizer> boxQuantizer;
    std::unique_ptr<CoordinatesQuantizer> coordinatesQuantizer;
    std::unordered_set<std::string> blackListOfPhraseGrounding;
};

} // namespace florence2