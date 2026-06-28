#pragma once

#include <string>
#include <cstdint>

namespace florence2 {

class IModelSource {
public:
    enum class Model {
        DecoderModelMerged,
        EmbedTokens,
        EncoderModel,
        VisionEncoder,
        DecoderModel
    };

    virtual ~IModelSource() = default;

    virtual bool try_get_model_path(Model model, std::string& model_path) = 0;
    virtual std::vector<uint8_t> get_model_bytes(Model model) = 0;
};

} // namespace florence2