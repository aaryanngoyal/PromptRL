#pragma once
#include "candidate.h"
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

enum class Result{
    FAILURE,
    ALREADY_EXISTS,
    SUCCESS,
    PARTIAL
};

using candidateManager = std::unordered_map<Tid, std::unique_ptr<Candidate>>;
using scoreMap = std::map<double,std::vector<Tid>,std::greater<double>>;

class CandidateManager{
    private:
        candidateManager candidateManager_;
        scoreMap scoreMap_;
    public:

        void addCandidateInScoreMap(Tid id, std::optional<double> score){
            if(score == std::nullopt){
                return;
            }
            scoreMap_[score.value()].emplace_back(id);
            return;
        }
        bool addCandidateInCandidateManager(std:: unique_ptr<Candidate> c){
            Tid id = c -> getId();
            candidateManager_[id] = std:: move(c);
            return true;
        }
        Result addCandidate(std:: unique_ptr<Candidate> c){
            bool candidateManagerSuccess = false;
            Tid id = c -> getId();
            std:: optional<double> score = c -> getScore();
            if(candidateManager_.contains(id)){
                return Result::ALREADY_EXISTS;
            }
            candidateManagerSuccess = addCandidateInCandidateManager(std::move(c));
            addCandidateInScoreMap(id, score);

            if(candidateManagerSuccess) return Result::SUCCESS;
            //Reserved for future scope of the given function
            return Result::FAILURE;
        }

};
