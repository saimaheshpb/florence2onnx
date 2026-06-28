#include "post_processing/post_processor.h"
#include <regex>
#include <algorithm>

namespace florence2
{

    Florence2PostProcessorConfig::Florence2PostProcessorConfig()
    {
        parseTasks = {
            {PostProcessingTypes::od, false},
            {PostProcessingTypes::ocr_with_region, false},
            {PostProcessingTypes::phrase_grounding, true},
            {PostProcessingTypes::pure_text, false},
            {PostProcessingTypes::description_with_bboxes, false},
            {PostProcessingTypes::description_with_polygons, false},
            {PostProcessingTypes::polygons, false},
            {PostProcessingTypes::bboxes, false},
            {PostProcessingTypes::description_with_bboxes_or_polygons, false}};
    }

    Florence2PostProcessor::Florence2PostProcessor()
    {
        for (const auto &task : config.parseTasks)
        {
            parseTaskConfigs[task.taskName] = task;
        }

        boxQuantizer = std::make_unique<BoxQuantizer>(
            config.boxQuantizationMode,
            std::make_pair(config.numBboxWidthBins, config.numBboxHeightBins));

        coordinatesQuantizer = std::make_unique<CoordinatesQuantizer>(
            config.coordinatesQuantizationMode,
            std::make_pair(config.coordinatesWidthBins, config.coordinatesHeightBins));

        // Initialize blacklist
        blackListOfPhraseGrounding = {
            "it", "I", "me", "mine",
            "you", "your", "yours",
            "he", "him", "his",
            "she", "her", "hers",
            "they", "them", "their", "theirs",
            "one", "oneself",
            // ... (Add all other blacklist items)
            "other objects", "lots", "a set"};
    }

    PostProcessingTypes Florence2PostProcessor::GetPostProcessingType(TaskType taskType)
    {
        switch (taskType)
        {
        case TaskType::OCR:
            return PostProcessingTypes::pure_text;
        case TaskType::OCR_WITH_REGION:
            return PostProcessingTypes::ocr_with_region;
        case TaskType::CAPTION:
        case TaskType::DETAILED_CAPTION:
        case TaskType::MORE_DETAILED_CAPTION:
            return PostProcessingTypes::pure_text;
        case TaskType::OD:
        case TaskType::DENSE_REGION_CAPTION:
            return PostProcessingTypes::description_with_bboxes;
        case TaskType::CAPTION_TO_PHRASE_GROUNDING:
            return PostProcessingTypes::phrase_grounding;
        case TaskType::REFERRING_EXPRESSION_SEGMENTATION:
        case TaskType::REGION_TO_SEGMENTATION:
            return PostProcessingTypes::polygons;
        case TaskType::OPEN_VOCABULARY_DETECTION:
            return PostProcessingTypes::description_with_bboxes_or_polygons;
        case TaskType::REGION_TO_CATEGORY:
        case TaskType::REGION_TO_DESCRIPTION:
        case TaskType::REGION_TO_OCR:
            return PostProcessingTypes::pure_text;
        case TaskType::REGION_PROPOSAL:
            return PostProcessingTypes::bboxes;
        default:
            return PostProcessingTypes::pure_text;
        }
    }

    std::string Florence2PostProcessor::ReplaceStartAndEndToken(const std::string &text)
    {
        std::string result = text;
        size_t pos;

        while ((pos = result.find("<s>")) != std::string::npos)
        {
            result.erase(pos, 3);
        }
        while ((pos = result.find("</s>")) != std::string::npos)
        {
            result.erase(pos, 4);
        }

        return result;
    }

    std::vector<LabeledOCRBox> Florence2PostProcessor::ParseOcrFromTextAndSpans(
        const std::string &text,
        std::pair<int, int> imageSize)
    {
        std::vector<LabeledOCRBox> instances;

        // Extract labels array
        std::regex labelPattern(R"('labels':\s*\[(.*?)\])");
        std::smatch labelMatch;
        if (!std::regex_search(text, labelMatch, labelPattern))
        {
            return instances;
        }

        // Parse individual labels
        std::string labelsStr = labelMatch[1];
        std::regex singleLabelPattern(R"('([^']*)')");
        std::vector<std::string> labels;

        std::string::const_iterator labelsStart(labelsStr.cbegin());
        std::smatch labelMatches;
        while (std::regex_search(labelsStart, labelsStr.cend(), labelMatches, singleLabelPattern))
        {
            labels.push_back(labelMatches[1]);
            labelsStart = labelMatches.suffix().first;
        }

        // Extract quad_boxes array
        std::regex coordPattern(R"('quad_boxes':\s*\[(.*?)\])");
        std::smatch coordMatch;
        if (!std::regex_search(text, coordMatch, coordPattern))
        {
            return instances;
        }

        // Parse individual coordinate arrays
        std::string coordsStr = coordMatch[1];
        std::regex arrayPattern(R"(\[([\d\., ]+)\])");
        std::string::const_iterator coordsStart(coordsStr.cbegin());
        std::smatch coordMatches;
        size_t labelIndex = 0;

        while (std::regex_search(coordsStart, coordsStr.cend(), coordMatches, arrayPattern))
        {
            if (labelIndex >= labels.size())
                break;

            LabeledOCRBox ocrBox;
            ocrBox.text = labels[labelIndex++];

            // Parse coordinates from the matched array
            std::string coordArray = coordMatches[1];
            std::regex numberPattern(R"([\d\.]+)");
            std::string::const_iterator numStart(coordArray.cbegin());
            std::smatch numMatch;
            std::vector<float> coords;

            while (std::regex_search(numStart, coordArray.cend(), numMatch, numberPattern))
            {
                coords.push_back(std::stof(numMatch[0]));
                numStart = numMatch.suffix().first;
            }

            // Convert coordinates to quad box points
            if (coords.size() >= 8)
            { // Ensure we have 4 points (8 coordinates)
                for (size_t i = 0; i < coords.size(); i += 2)
                {
                    ocrBox.quad_box.push_back(Coordinates<float>(coords[i], coords[i + 1]));
                }
                instances.push_back(ocrBox);
            }

            coordsStart = coordMatches.suffix().first;
        }

        return instances;
    }

    std::vector<LabeledBoundingBoxes> Florence2PostProcessor::ParsePhraseGroundingFromTextAndSpans(
        const std::string &text,
        std::pair<int, int> imageSize)
    {
        std::vector<LabeledBoundingBoxes> results;
        std::string processedText = text;

        // Remove start/end tokens and pad
        processedText = std::regex_replace(processedText, std::regex("<s>|</s>|<pad>"), "");

        // Match phrases with locations
        std::regex phrasePattern(R"([^<]+(?:<loc_\d+>){4,})");
        std::regex textPattern(R"(^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_))");
        std::regex boxPattern(R"(<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>)");

        std::string::const_iterator searchStart(processedText.cbegin());
        std::smatch phraseMatches;

        while (std::regex_search(searchStart, processedText.cend(), phraseMatches, phrasePattern))
        {
            std::string phraseText = phraseMatches[0].str();

            // Remove ground and obj tags
            phraseText = std::regex_replace(phraseText, std::regex("<ground>|<obj>", std::regex_constants::icase), "");

            if (phraseText.empty())
            {
                searchStart = phraseMatches.suffix().first;
                continue;
            }

            std::smatch textMatch;
            if (!std::regex_search(phraseText, textMatch, textPattern))
            {
                searchStart = phraseMatches.suffix().first;
                continue;
            }

            std::string phrase = textMatch[1].str();
            phrase = std::regex_replace(phrase, std::regex("\\s+"), " ");
            phrase = std::regex_replace(phrase, std::regex("^\\s+|\\s+$"), "");

            if (blackListOfPhraseGrounding.find(phrase) != blackListOfPhraseGrounding.end())
            {
                searchStart = phraseMatches.suffix().first;
                continue;
            }

            std::vector<BoundingBox<int>> bboxBins;
            std::string::const_iterator boxStart(phraseText.cbegin());
            std::smatch boxMatch;

            while (std::regex_search(boxStart, phraseText.cend(), boxMatch, boxPattern))
            {
                std::vector<int> coords;
                for (int i = 1; i <= 4; i++)
                {
                    coords.push_back(std::stoi(boxMatch[i].str()));
                }
                bboxBins.push_back(BoundingBox<int>(coords));
                boxStart = boxMatch.suffix().first;
            }

            if (!bboxBins.empty())
            {
                LabeledBoundingBoxes box;
                box.bboxes = boxQuantizer->dequantize(bboxBins, imageSize);

                // Filter non-ASCII characters
                std::string filtered;
                std::copy_if(phrase.begin(), phrase.end(), std::back_inserter(filtered),
                             [](char c)
                             { return static_cast<unsigned char>(c) < 128; });

                box.label = ReplaceStartAndEndToken(filtered);
                results.push_back(box);
            }

            searchStart = phraseMatches.suffix().first;
        }

        return results;
    }

    std::vector<LabeledBoundingBoxes> Florence2PostProcessor::ParseDescriptionWithBboxesFromTextAndSpans(
        const std::string &text,
        std::pair<int, int> imageSize,
        bool allowEmptyPhrase)
    {
        std::vector<LabeledBoundingBoxes> results;
        std::string processedText = text;

        // Remove start/end tokens and pad
        processedText = std::regex_replace(processedText, std::regex("<s>|</s>|<pad>"), "");

        // Pattern for phrases with or without text depending on allowEmptyPhrase
        std::regex phrasePattern = allowEmptyPhrase ? std::regex(R"((?:(?:<loc_\d+>){4,}))") : std::regex(R"([^<]+(?:<loc_\d+>){4,})");

        // Pattern for extracting text and boxes
        std::regex textPattern(R"(^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_))");
        std::regex boxPattern(R"(<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>)");

        std::string::const_iterator searchStart(processedText.cbegin());
        std::smatch phraseMatches;

        while (std::regex_search(searchStart, processedText.cend(), phraseMatches, phrasePattern))
        {
            std::string phraseText = phraseMatches[0].str();

            // Remove specific tags
            phraseText = std::regex_replace(phraseText, std::regex("<ground>|<obj>", std::regex_constants::icase), "");

            if (phraseText.empty() && !allowEmptyPhrase)
            {
                searchStart = phraseMatches.suffix().first;
                continue;
            }

            // Extract phrase text
            std::smatch textMatch;
            std::string phrase;
            if (std::regex_search(phraseText, textMatch, textPattern))
            {
                phrase = textMatch[1].str();
                phrase = std::regex_replace(phrase, std::regex("\\s+"), " ");
                phrase = std::regex_replace(phrase, std::regex("^\\s+|\\s+$"), "");
            }

            // Extract bounding boxes
            std::vector<BoundingBox<int>> bboxBins;
            std::string::const_iterator boxStart(phraseText.cbegin());
            std::smatch boxMatch;

            while (std::regex_search(boxStart, phraseText.cend(), boxMatch, boxPattern))
            {
                std::vector<int> coords;
                for (int i = 1; i <= 4; i++)
                {
                    coords.push_back(std::stoi(boxMatch[i].str()));
                }
                bboxBins.push_back(BoundingBox<int>(coords));
                boxStart = boxMatch.suffix().first;
            }

            if (!bboxBins.empty())
            {
                LabeledBoundingBoxes box;
                box.bboxes = boxQuantizer->dequantize(bboxBins, imageSize);

                // Filter non-ASCII characters
                std::string filtered;
                std::copy_if(phrase.begin(), phrase.end(), std::back_inserter(filtered),
                             [](char c)
                             { return static_cast<unsigned char>(c) < 128; });

                box.label = ReplaceStartAndEndToken(filtered);
                results.push_back(box);
            }

            searchStart = phraseMatches.suffix().first;
        }

        return results;
    }

    std::vector<LabeledPolygon> Florence2PostProcessor::ParseDescriptionWithPolygonsFromTextAndSpans(
        const std::string &text,
        std::pair<int, int> imageSize,
        bool allowEmptyPhrase,
        const std::string &polygonSepToken,
        const std::string &polygonStartToken,
        const std::string &polygonEndToken,
        bool withBoxAtStart)
    {
        std::vector<LabeledPolygon> results;
        std::string processedText = text;

        // Remove start/end tokens and pad
        processedText = std::regex_replace(processedText, std::regex("<s>|</s>|<pad>"), "");

        // Create pattern for phrases
        std::string phrasePatternStr = allowEmptyPhrase ? "(?:(?:<loc_\\d+>|" + std::regex_replace(polygonSepToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)") +
                                                              "|" + std::regex_replace(polygonStartToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)") +
                                                              "|" + std::regex_replace(polygonEndToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)") + "){4,})"
                                                        : "([^<]+(?:<loc_\\d+>|" + std::regex_replace(polygonSepToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)") +
                                                              "|" + std::regex_replace(polygonStartToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)") +
                                                              "|" + std::regex_replace(polygonEndToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)") + "){4,})";

        std::regex phrasePattern(phrasePatternStr);
        std::regex phraseStringPattern(R"(^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_|<poly>))");
        std::regex boxPattern(R"(((?:<loc_\d+>)+)(?:)" + std::regex_replace(polygonSepToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)") + R"(|$))");
        std::regex polygonsInstancePattern(std::regex_replace(polygonStartToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)") +
                                           R"((.*?))" +
                                           std::regex_replace(polygonEndToken, std::regex(R"([.^$*+?()[{\|])"), R"(\$&)"));

        std::string::const_iterator searchStart(processedText.cbegin());
        std::smatch phraseMatches;

        while (std::regex_search(searchStart, processedText.cend(), phraseMatches, phrasePattern))
        {
            LabeledPolygon polygon;
            std::string phraseText = phraseMatches[0].str();

            if (phraseText.empty() && !allowEmptyPhrase)
            {
                searchStart = phraseMatches.suffix().first;
                continue;
            }

            // Extract phrase
            std::smatch phraseMatch;
            if (std::regex_search(phraseText, phraseMatch, phraseStringPattern))
            {
                polygon.label = phraseMatch[1].str();
            }

            // Process polygon instances
            std::vector<std::string> polygonsInstancesParsed;
            if (phraseText.find(polygonStartToken) != std::string::npos &&
                phraseText.find(polygonEndToken) != std::string::npos)
            {
                std::string::const_iterator polyStart(phraseText.cbegin());
                std::smatch polyMatch;
                while (std::regex_search(polyStart, phraseText.cend(), polyMatch, polygonsInstancePattern))
                {
                    polygonsInstancesParsed.push_back(polyMatch[1].str());
                    polyStart = polyMatch.suffix().first;
                }
            }
            else
            {
                polygonsInstancesParsed.push_back(phraseText);
            }

            // Process each polygon instance
            for (const auto &polygonsInstance : polygonsInstancesParsed)
            {
                std::vector<Coordinates<float>> fullPolygon;
                std::vector<BoundingBox<float>> bboxes;

                std::string::const_iterator boxStart(polygonsInstance.cbegin());
                std::smatch boxMatch;

                while (std::regex_search(boxStart, polygonsInstance.cend(), boxMatch, boxPattern))
                {
                    std::regex coordPattern(R"(<loc_(\d+)>)");
                    std::string coords = boxMatch[1].str();

                    std::vector<int> polygon_coords;
                    std::string::const_iterator coordStart(coords.cbegin());
                    std::smatch coordMatch;

                    while (std::regex_search(coordStart, coords.cend(), coordMatch, coordPattern))
                    {
                        polygon_coords.push_back(std::stoi(coordMatch[1].str()));
                        coordStart = coordMatch.suffix().first;
                    }

                    if (withBoxAtStart && bboxes.empty())
                    {
                        if (polygon_coords.size() >= 4)
                        {
                            std::vector<int> bbox_coords(polygon_coords.begin(), polygon_coords.begin() + 4);
                            auto dequantized_bbox = boxQuantizer->dequantize({BoundingBox<int>(bbox_coords)}, imageSize);
                            bboxes.insert(bboxes.end(), dequantized_bbox.begin(), dequantized_bbox.end());
                            polygon_coords.erase(polygon_coords.begin(), polygon_coords.begin() + 4);
                        }
                        else
                        {
                            bboxes.push_back(BoundingBox<float>(0, 0, 0, 0));
                        }
                    }

                    // Process remaining coordinates as polygon points
                    if (polygon_coords.size() % 2 == 1)
                    {
                        polygon_coords.pop_back(); // Remove unpaired coordinate
                    }

                    std::vector<Coordinates<int>> polygon_points;
                    for (size_t i = 0; i < polygon_coords.size(); i += 2)
                    {
                        polygon_points.push_back(Coordinates<int>(polygon_coords[i], polygon_coords[i + 1]));
                    }

                    auto dequantized_points = coordinatesQuantizer->dequantize(polygon_points, imageSize);
                    fullPolygon.insert(fullPolygon.end(), dequantized_points.begin(), dequantized_points.end());

                    boxStart = boxMatch.suffix().first;
                }

                if (!fullPolygon.empty())
                {
                    polygon.polygon = fullPolygon;
                    polygon.bboxes = bboxes;
                    results.push_back(polygon);
                }
            }

            searchStart = phraseMatches.suffix().first;
        }

        return results;
    }

    FlorenceResults Florence2PostProcessor::PostProcessGeneration(
        const std::string &text,
        TaskType task,
        std::pair<int, int> imageSize)
    {
        PostProcessingTypes postProcessingTask = GetPostProcessingType(task);
        FlorenceResults results;

        switch (postProcessingTask)
        {
        case PostProcessingTypes::pure_text:
        {
            results.pure_text = ReplaceStartAndEndToken(text);
            break;
        }
        case PostProcessingTypes::ocr_with_region:
        {
            results.ocr_bbox = ParseOcrFromTextAndSpans(text, imageSize);
            break;
        }
        case PostProcessingTypes::od:
        case PostProcessingTypes::bboxes:
        case PostProcessingTypes::description_with_bboxes:
        {
            results.bounding_boxes = ParseDescriptionWithBboxesFromTextAndSpans(
                text, imageSize, false);
            break;
        }
        case PostProcessingTypes::phrase_grounding:
        {
            results.bounding_boxes = ParsePhraseGroundingFromTextAndSpans(
                text, imageSize);
            break;
        }
        case PostProcessingTypes::description_with_polygons:
        {
            results.polygons = ParseDescriptionWithPolygonsFromTextAndSpans(
                text, imageSize, false);
            break;
        }
        case PostProcessingTypes::polygons:
        {
            results.polygons = ParseDescriptionWithPolygonsFromTextAndSpans(
                text, imageSize, true);
            break;
        }
        case PostProcessingTypes::description_with_bboxes_or_polygons:
        {
            if (text.find("<poly>") != std::string::npos)
            {
                results.polygons = ParseDescriptionWithPolygonsFromTextAndSpans(
                    text, imageSize, false);
            }
            else
            {
                results.bounding_boxes = ParseDescriptionWithBboxesFromTextAndSpans(
                    text, imageSize, false);
            }
            break;
        }
        default:
        {
            throw std::invalid_argument("Unknown task answer post processing type: " +
                                        std::to_string(static_cast<int>(postProcessingTask)));
        }
        }

        return results;
    }

}

// Would you like me to continue with the implementation of the other methods?
// The remaining methods would be:
// - ParseOcrFromTextAndSpans
// - ParsePhraseGroundingFromTextAndSpans
// - ParseDescriptionWithBboxesFromTextAndSpans
// - ParseDescriptionWithPolygonsFromTextAndSpans
// - PostProcessGeneration