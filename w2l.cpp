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

#define W2LOG(LEVEL) (LOG(INFO) << "[w2lapi] ")

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
        LOG(FATAL) << "Getting text directly from an emission is not currently supported\n";
        // auto viterbiPath = afToVector<int>(engine->criterion->viterbiPath(emission.array()));
        // if (engine->criterionType == kCtcCriterion || engine->criterionType == kAsgCriterion) {
        //     uniq(viterbiPath);
        // }
        // remapLabels(viterbiPath, engine->tokenDict);
        // auto letters = tknPrediction2Ltr(viterbiPath, engine->tokenDict);
        // if (letters.size() > 0) {
        //     std::string str = tensor2letters(viterbiPath, engine->tokenDict);
        //     return strdup(str.c_str());
        // }
        // return strdup("");
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
    WrapDecoder(Engine *engine, const char *languageModelPath, const char *lexiconPath) : engine(engine){
        W2LOG(INFO) << "Initializing decoder";
        auto lexicon = loadWords(lexiconPath, -1);
        W2LOG(INFO) << "Allocated leixcon";
        wordDict = createWordDict(lexicon);
        W2LOG(INFO) << "Allocated word dict";
        lm = std::make_shared<KenLM>(languageModelPath, wordDict);
        W2LOG(INFO) << "Allocated language model";

        int silIdx = engine->tokenDict.getIndex(kSilToken);
        W2LOG(INFO) << "SIL idx: " << silIdx;
        int blankIdx = engine->criterionType == kCtcCriterion ? engine->tokenDict.getIndex(kBlankToken) : -1;
        W2LOG(INFO) << "Blank idx: " << blankIdx;
        int unkIdx = wordDict.getIndex(kUnkToken);
        W2LOG(INFO) << "indexs:" << silIdx << " " << blankIdx << " " << unkIdx;
        trie = std::make_shared<Trie>(engine->tokenDict.indexSize(), silIdx);
        W2LOG(INFO) << "Allocated token trie";

        auto start_state = lm->start(false);
        for (auto& it : lexicon) {
            std::string word = it.first;
            int usrIdx = wordDict.getIndex(word);
            float score;
            LMStatePtr dummyState;
            // if (usrIdx == unkIdx) { // We don't insert unknown words
            //     continue;
            // }
            std::tie(dummyState, score) = lm->score(start_state, usrIdx);
            for (auto& tokens : it.second) {
                auto tokensTensor = tkn2Idx(tokens, engine->tokenDict, FLAGS_replabel);
                trie->insert(tokensTensor, usrIdx, score);
            }
        }

        W2LOG(INFO) << "Trie planted";

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
        W2LOG(INFO) << "Trie smear completed";

        CriterionType criterionType = CriterionType::ASG;
        if (FLAGS_criterion == kCtcCriterion) {
          criterionType = CriterionType::CTC;
        } else if (FLAGS_criterion == kSeq2SeqCriterion) {
          criterionType = CriterionType::S2S;
        } else if (FLAGS_criterion != kAsgCriterion) {
          W2LOG(INFO) << "Invalid model type: " << FLAGS_criterion;
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

        W2LOG(INFO) << "Loaded decoder options";

        auto transition = afToVector<float>(engine->criterion->param(0).array());

        decoder.reset(new WordLMDecoder(
            decoderOpt,
            trie,
            lm,
            silIdx,
            blankIdx,
            unkIdx,
            transition));

        W2LOG(INFO) << "Decoder initialized";
    }
    ~WrapDecoder() {}

    char *decode(Emission *emission) {
        auto rawEmission = emission->emission;
        auto emissionVec = afToVector<float>(rawEmission);
        int N = rawEmission.dims(0);
        int T = rawEmission.dims(1);

        auto data = afToVector<float>(emission->emission.array());
        auto results = decoder->decode(data.data(), T, N);

        // Cleanup predictions
        auto& rawWordPrediction = results[0].words;
        auto& rawTokenPrediction = results[0].tokens;

        // auto letterTarget = tknTarget2Ltr(tokenTarget, engine->tokenDict);
        auto letterPrediction =
            tknPrediction2Ltr(rawTokenPrediction, engine->tokenDict);

        W2LOG(INFO) << "|p|: " << join(" ", letterPrediction) << std::endl;

        std::vector<std::string> wordPrediction;
        rawWordPrediction =
            validateIdx(rawWordPrediction, wordDict.getIndex(kUnkToken));
        wordPrediction = wrdIdx2Wrd(rawWordPrediction, wordDict);

        auto words = join(" ", wordPrediction);

        return strdup(words.c_str());
    }

    std::shared_ptr<KenLM> lm;
    std::unique_ptr<Decoder> decoder;
    std::shared_ptr<Trie> trie;
    Engine * engine;
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
