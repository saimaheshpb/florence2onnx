#pragma once

#include <string>
#include <optional>

namespace florence2 {

class Tokens {
public:
    virtual ~Tokens() = default;

    // Move constructor to public section and make it public
    Tokens(const std::string& padding,
          const std::string& unknown,
          const std::string& classification,
          const std::string& separation,
          const std::string& mask,
          const std::optional<std::string>& eos,
          const std::optional<std::string>& bos)
        : padding_(padding)
        , unknown_(unknown)
        , classification_(classification)
        , separation_(separation)
        , mask_(mask)
        , end_of_sequence_(eos)
        , beginning_of_sequence_(bos) {}

    // Getter methods for token strings
    virtual std::string padding() const { return padding_; }
    virtual std::string unknown() const { return unknown_; }
    virtual std::string classification() const { return classification_; }
    virtual std::string separation() const { return separation_; }
    virtual std::string mask() const { return mask_; }
    virtual std::optional<std::string> end_of_sequence() const { return end_of_sequence_; }
    virtual std::optional<std::string> beginning_of_sequence() const { return beginning_of_sequence_; }

private:
    std::string padding_;
    std::string unknown_;
    std::string classification_;
    std::string separation_;
    std::string mask_;
    std::optional<std::string> end_of_sequence_;
    std::optional<std::string> beginning_of_sequence_;
};

class SentenceTransformerTokens : public Tokens {
public:
    SentenceTransformerTokens()
        : Tokens("",              // padding
                "[UNK]",          // unknown
                "[CLS]",          // classification
                "[SEP]",          // separation
                "[MASK]",         // mask
                std::nullopt,     // end_of_sequence
                std::nullopt)     // beginning_of_sequence
    {}

    // Override getters to ensure they're marked override and const
    std::string padding() const override { return ""; }
    std::string unknown() const override { return "[UNK]"; }
    std::string classification() const override { return "[CLS]"; }
    std::string separation() const override { return "[SEP]"; }
    std::string mask() const override { return "[MASK]"; }
    std::optional<std::string> end_of_sequence() const override { return std::nullopt; }
    std::optional<std::string> beginning_of_sequence() const override { return std::nullopt; }
};

} // namespace florence2