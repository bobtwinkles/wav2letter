#include <stdlib.h>
#include <string>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/W2lDataset.h"
#include "module/module.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"
#include "decoder/Decoder.h"
#include "lm/KenLM.h"
#include "decoder/Trie.h"
#include "decoder/WordLMDecoder.h"

using namespace w2l;

class EngineBase {
public:
    EngineBase(const char *tokensPath) : tokenDict(tokensPath) {}

    int numClasses;
    std::unordered_map<std::string, std::string> config;
    std::shared_ptr<fl::Module> network;
    std::shared_ptr<SequenceCriterion> criterion;
    std::string criterionType;
    Dictionary tokenDict;
};

class Emission {
public:
    Emission(EngineBase *engine, fl::Variable emission) {
        this->engine = engine;
        this->emission = emission;
    }
    ~Emission() {}

    char *text() {
        auto viterbiPath = afToVector<int>(engine->criterion->viterbiPath(emission.array()));
        if (engine->criterionType == kCtcCriterion || engine->criterionType == kAsgCriterion) {
            uniq(viterbiPath);
        }
        remapLabels(viterbiPath, engine->tokenDict);
        auto letters = letters(viterbiPath, engine->tokenDict);
        if (letters.size() > 0) {
            std::string str = tensor2letters(viterbiPath, engine->tokenDict);
            return strdup(str.c_str());
        }
        return strdup("");
    }

    EngineBase *engine;
    fl::Variable emission;
};

class Engine : public EngineBase {
public:
    Engine(const char *acousticModelPath, const char *tokensPath) : EngineBase(tokensPath) {
        // TODO: set criterionType "correctly"
        W2lSerializer::load(acousticModelPath, config, network, criterion);
        auto flags = config.find(kGflags);
        // loading flags globally like this is gross, only way to work around it will be parameterizing everything about wav2letter better
        gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

        criterionType = FLAGS_criterion;
        network->eval();
        criterion->eval();

        tokenDict = Dictionary(tokensPath);
        numClasses = tokenDict.indexSize();
    }
    ~Engine() {}

    Emission *process(float *samples, size_t sample_count) {
        struct W2lLoaderData data = {};
        std::copy(samples, samples + sample_count, std::back_inserter(data.input));

        auto feat = featurize({data}, {});
        auto result = af::array(feat.inputDims, feat.input.data());
        auto rawEmission = network->forward({fl::input(result)}).front();
        return new Emission(this, rawEmission);
    }
};

class WrapDecoder {
public:
    WrapDecoder(Engine *engine, const char *languageModelPath, const char *lexiconPath) {
        std::cout << 1 << std::endl;
        auto lexicon = loadWords(lexiconPath, -1);
        wordDict = createWordDict(lexicon);
        std::cout << 2 << std::endl;

        int silIdx = engine->tokenDict.getIndex(kSilToken);
        int blankIdx = engine->criterionType == kCtcCriterion ? engine->tokenDict.getIndex(kBlankToken) : -1;
        int unkIdx = wordDict.getIndex(kUnkToken);
        trie = std::make_shared<Trie>(engine->tokenDict.indexSize(), silIdx);
        auto start_state = lm->start(false);
        for (auto& it : lexicon) {
            std::string word = it.first;
            int usrIdx = wordDict.getIndex(word);
            float score;
            LMStatePtr dummyState;
            // if (lmIdx == unkIdx) { // We don't insert unknown words
            //     continue;
            // }
            std::tie(dummyState, score) = lm->score(start_state, usrIdx);
            for (auto& tokens : it.second) {
                auto tokensTensor = tkn2Idx(tokens, engine->tokenDict, FLAGS_replabel);
                trie->insert(
                        tokensTensor,
                        wordDict.getIndex(word),
                        score);
            }
        }
        std::cout << 3 << std::endl;

        SmearingMode smear_mode = SmearingMode::LOGADD;
        // TODO: smear mode argument?
        /*
        SmearingMode smear_mode = SmearingMode::NONE;
        if (FLAGS_smearing == "logadd") {
            smear_mode = SmearingMode::LOGADD;
        } else if (FLAGS_smearing == "max") {
            smear_mode = SmearingMode::MAX;
        } else if (FLAGS_smearing != "none") {
            LOG(FATAL) << "[Decoder] Invalid smearing mode: " << FLAGS_smearing;
        }
        */
        trie->smear(smear_mode);
        std::cout << 4 << std::endl;

        std::cout << 5 << std::endl;
        lm = std::make_shared<KenLM>(languageModelPath, engine->tokenDict);

        std::cout << 6 << std::endl;

        CriterionType criterionType = CriterionType::ASG;
        if (FLAGS_criterion == kCtcCriterion) {
          criterionType = CriterionType::CTC;
        } else if (FLAGS_criterion == kSeq2SeqCriterion) {
          criterionType = CriterionType::S2S;
        } else if (FLAGS_criterion != kAsgCriterion) {
          LOG(FATAL) << "[Decoder] Invalid model type: " << FLAGS_criterion;
        }


        // FIXME, don't use global flags
        decoderOpt = DecoderOptions(
            FLAGS_beamsize,
            static_cast<float>(FLAGS_beamthreshold),
            static_cast<float>(FLAGS_lmweight),
            static_cast<float>(FLAGS_wordscore),
            static_cast<float>(FLAGS_unkweight),
            FLAGS_logadd,
            static_cast<float>(FLAGS_silweight),
            criterionType);

        LOG(INFO) << "[w2lapi] Loaded decoder options";
    }
    ~WrapDecoder() {}

    char *decode(Emission *emission) {
        auto transition = afToVector<float>(emission->engine->criterion->param(0).array());
        auto rawEmission = emission->emission;
        auto emissionVec = afToVector<float>(rawEmission);
        int N = rawEmission.dims(0);
        int T = rawEmission.dims(1);

        w2l::WordLMDecoder decoder(
            decoderOpt,
            trie,
            lm,
            silIdx,
            blankIdx,
            unkIdx,
            transition);

        LOG(INFO) << "[w2lapi] Created WordLMDecdoer";

        auto data = afToVector<float>(emission->emission.array());
        auto results = decoder.decode(data.data(), T, N);

        // Cleanup predictions
        auto& rawWordPrediction = results[0].words;
        auto& rawTokenPrediction = results[0].tokens;

        auto letterPrediction =
            tknPrediction2Ltr(rawTokenPrediction, emission->engine->tokenDict);
        std::vector<std::string> wordPrediction;
        if (!FLAGS_lexicon.empty() && FLAGS_criterion != kSeq2SeqCriterion) {
          rawWordPrediction =
              validateIdx(rawWordPrediction, wordDict.getIndex(kUnkToken));
          wordPrediction = wrdIdx2Wrd(rawWordPrediction, wordDict);
        } else {
          wordPrediction = tkn2Wrd(letterPrediction);
        }

        auto words = join(" ", wordPrediction);

        return strdup(words.c_str());
    }

    std::shared_ptr<KenLM> lm;
    std::shared_ptr<Trie> trie;
    int silIdx;
    int blankIdx;
    int unkIdx;
    Dictionary wordDict;
    DecoderOptions decoderOpt;
};

extern "C" {

#include "w2l.h"

typedef struct w2l_engine w2l_engine;
typedef struct w2l_decoder w2l_decoder;
typedef struct w2l_emission w2l_emission;

w2l_engine *w2l_engine_new(const char *acoustic_model_path, const char *tokens_path) {
    // TODO: what other engine config do I need?
    auto engine = new Engine(acoustic_model_path, tokens_path);
    return reinterpret_cast<w2l_engine *>(engine);
}

w2l_emission *w2l_engine_process(w2l_engine *engine, float *samples, size_t sample_count) {
    auto emission = reinterpret_cast<Engine *>(engine)->process(samples, sample_count);
    return reinterpret_cast<w2l_emission *>(emission);
}

void w2l_engine_free(w2l_engine *engine) {
    if (engine)
        delete reinterpret_cast<Engine *>(engine);
}

char *w2l_emission_text(w2l_emission *emission) {
    // TODO: I think w2l_emission needs a pointer to the criterion to do viterbiPath
    //       I could just use a shared_ptr to just the criterion and not point emission -> engine
    //       so I'm not passing a raw shared_ptr back from C the api
    // TODO: do a viterbiPath here
    return reinterpret_cast<Emission *>(emission)->text();
}

void w2l_emission_free(w2l_emission *emission) {
    if (emission)
        delete reinterpret_cast<Emission *>(emission);
}

w2l_decoder *w2l_decoder_new(w2l_engine *engine, const char *kenlm_model_path, const char *lexicon_path) {
    // TODO: what other config? beam size? smearing? lm weight?
    auto decoder = new WrapDecoder(reinterpret_cast<Engine *>(engine), kenlm_model_path, lexicon_path);
    return reinterpret_cast<w2l_decoder *>(decoder);
}

char *w2l_decoder_decode(w2l_decoder *decoder, w2l_emission *emission) {
    return reinterpret_cast<WrapDecoder *>(decoder)->decode(reinterpret_cast<Emission *>(emission));
}

void w2l_decoder_free(w2l_decoder *decoder) {
    if (decoder)
        delete reinterpret_cast<WrapDecoder *>(decoder);
}

} // extern "C"
