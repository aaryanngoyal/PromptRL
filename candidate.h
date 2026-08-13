# pragma once
#include <cstdint>
#include <iostream>
#include <optional>
#include <utility>

using Tid = std::uint64_t;

enum class Action{
    ADD_CLARIFIER,
    ADD_EXAMPLE,
    REPHRASE,
    SIMPLIFY
};

class Candidate{

    private:
    const Tid CandidateId_;
    const std:: string prompt_;
    std::optional<double> score_;


    public:
    Candidate(Tid CandidateId,std:: string prompt, std:: optional<double> score): CandidateId_(CandidateId),prompt_(std:: move(prompt)), score_(score) {}

    //Public API's
    Tid getId() const {return CandidateId_;}

    const std::string& getPrompt() const {return prompt_;}
    const std::optional<double> getScore() const {return score_;}

    bool setScore(double score) {
        if(score_.has_value()){
            return false;
        }
        else{
            score_ = score;
            return true;
        }
    }

};
