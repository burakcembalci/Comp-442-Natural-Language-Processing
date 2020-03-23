#imports
import Pkg
using Pkg; for p in ("Knet","IterTools","WordTokenizers","Test","Random","Statistics","Dates","LinearAlgebra","CuArrays"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Statistics, IterTools, WordTokenizers, Test, Knet, Random, Dates, Base.Iterators, LinearAlgebra

# Update and list all packages
Pkg.update()
pkgs = Pkg.installed()

for package in keys(pkgs)
    if pkgs[package] == nothing
        pkgs[package] = VersionNumber("0.0.1")
    end
    println("Package name: ", package, " Version: ", pkgs[package])
end

using CuArrays: CuArrays, usage_limit
CuArrays.usage_limit[] = 8_000_000_000
BATCH_SIZE = 64

Knet.atype() = KnetArray{Float32} #Array{Float32}
is_lstm_strategy_on = true # if true rnn type becomes lstm, otherwise we preferred to use relu
gpu() # GPU test must result as 0

# Vocabulary Structure
struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer    
end

function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    vocab_freq = Dict{String,Int64}(unk => 0, eos => 0)
    w2i = Dict{String, Int64}(unk => 2, eos => 1)
    i2w = Vector{String}()
    tags = Vector{String}()

    push!(i2w, eos)
    push!(i2w, unk)
        
    open(file) do f
        for line in eachline(f)
            tag, sentence = split(strip(lowercase(line))," ||| ")
            !(tag in tags) && push!(tags, tag)
            
            sentence = tokenizer(sentence, [' '], keepempty = false)
            
            for word in sentence
                word == unk && continue
                word == eos && continue # They are default ones to be added later
                vocab_freq[word] = get!(vocab_freq, word, 0) + 1
            end
        end
        close(f)
    end


    # End of vanilla implementation of the vocabulary
    # From here we must add the mincount and vocabsize properties
    # We must change the first two property of the vocab wrt those paramaters
    vocab_freq = sort!(
        collect(vocab_freq),
        by = tuple -> last(tuple),
        rev = true,
    )

    if length(vocab_freq)>vocabsize - 2 # eos and unk ones
        vocab_freq = vocab_freq[1:vocabsize-2] # trim to fit the size
    end

    #vocab_freq = reverse(vocab_freq)

    while true
        length(vocab_freq)==0 && break
        word,freq = vocab_freq[end]
        freq>=mincount && break # since it is already ordered
        vocab_freq = vocab_freq[1:(end - 1)]
    end
    #pushfirst!(vocab_freq,unk=>1,eos=>1) # freq does not matter, just adding the
    for i in 1:length(vocab_freq)
        word, freq = vocab_freq[i]
        ind = (get!(w2i, word, 1+length(w2i)))
        (length(i2w) < ind) && push!(i2w, word)
    end

    return Vocab(w2i, i2w, tags, 2, 1, tokenizer)
end

# Special reader for the task
struct TextReader
    file::String
    vocab::Vocab
end

word2ind(dict,x) = get(dict, x, 2)
findtagid(tag,vector) = findall(x -> x == tag,vector)[1]

#Implementing the iterate function
function Base.iterate(r::TextReader, s=nothing)
    if s == nothing
        state = open(r.file)
        Base.iterate(r,state)
    else
        if eof(s) == true
            close(s)
            return nothing
        else
            line = readline(s)
            sent_ind = Int[]
            
            # Tagification
            tag, sentence = split(strip(lowercase(line))," ||| ")
            
            tagind = findtagid(tag,r.vocab.tags)
            push!(sent_ind,tagind)
            
            sent = r.vocab.tokenizer(strip(lowercase(sentence)), [' '], keepempty = false)
            
            for word in sent
                ind = word2ind(r.vocab.w2i,word)
                push!(sent_ind,ind)
            end
            push!(sent_ind,r.vocab.eos)
            return (sent_ind, s)
        end
    end
end


Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}


# File 
const datadir = "nn4nlp-code/data/classes"
isdir(datadir) || run(`git clone https://github.com/neubig/nn4nlp-code.git`)

if !isdefined(Main, :vocab)
    vocab = Vocab("$datadir/train.txt", mincount=2)

    train = TextReader("$datadir/train.txt", vocab)
    test = TextReader("$datadir/test.txt", vocab)

end

# Minibatching
struct LMData
    src::TextReader
    batchsize::Int
    maxlength::Int
    bucketwidth::Int
    buckets
end

function LMData(src::TextReader; batchsize = BATCH_SIZE, maxlength = typemax(Int), bucketwidth = 10)
    numbuckets = min(128, maxlength ÷ bucketwidth)
    buckets = [ [] for i in 1:numbuckets ]
    LMData(src, batchsize, maxlength, bucketwidth, buckets)
end

Base.IteratorSize(::Type{LMData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{LMData}) = Base.HasEltype()
Base.eltype(::Type{LMData}) = Matrix{Int}

function Base.iterate(d::LMData, state=nothing)
    if state == nothing
        for b in d.buckets; empty!(b); end
    end
    bucket,ibucket = nothing,nothing
    while true
        iter = (state === nothing ? iterate(d.src) : iterate(d.src, state))
        if iter === nothing
            ibucket = findfirst(x -> !isempty(x), d.buckets)
            bucket = (ibucket === nothing ? nothing : d.buckets[ibucket])
            break
        else
            sent, state = iter
            if length(sent) > d.maxlength || length(sent) == 0; continue; end
            ibucket = min(1 + (length(sent)-1) ÷ d.bucketwidth, length(d.buckets))
            bucket = d.buckets[ibucket]
            push!(bucket, sent)
            if length(bucket) === d.batchsize; break; end
        end
    end
    if bucket === nothing; return nothing; end
    batchsize = length(bucket)
    maxlen = maximum(length.(bucket))
    batch = fill(d.src.vocab.eos, batchsize, maxlen + 1)
    for i in 1:batchsize
        batch[i, 1:length(bucket[i])] = bucket[i]
    end
    empty!(bucket)
    return batch, state
end

# Mask!
function mask!(a,pad)
    matr = a
    for j in 1:size(matr)[1]
        i=0
        while i<(length(matr[j,:])-1)
            matr[j,length(matr[j,:])-i-1]!=pad && break

            if matr[j,length(matr[j,:])-i]== pad
                matr[j,length(matr[j,:])-i]= 0
            end
            i+=1
        end
    end
    matr
end

# Embed format updated with respect to task
struct Embed; w; end

function Embed(tagsize::Int, embedsize::Int)
    Embed(param(embedsize,tagsize))
end

function (l::Embed)(x)
    l.w[:,x]
end

#Linear
struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize,inputsize), param0(outputsize))
end

function (l::Linear)(x)
    l.w * mat(x,dims=1) .+ l.b
end

struct RNNsent_model
    embed::Embed        # language embedding
    rnn::RNN            # RNN (can be bidirectional)
    projection::Linear  # converts output to vocab scores
    dropout::Real       # dropout probability to prevent overfitting
    vocab::Vocab        # language vocabulary
end

function RNNsent_model(hidden::Int,      # hidden size for both the encoder and decoder RNN
                embsz::Int,          # embedding size
                vocab::Vocab;        # language vocabulary
                layers=1,            # number of layers
                bidirectional=false, # whether encoder RNN is bidirectional
                dropout=0)           # dropout probability

    embed = Embed(length(vocab.i2w),embsz)

    rnn = RNN(embsz,hidden;rnnType=is_lstm_strategy_on ? :lstm : :relu, numLayers=layers,bidirectional=bidirectional, dropout= dropout)
    
    layerMultiplier = bidirectional ? 2 : 1
    
    projection = Linear(layerMultiplier*hidden,length(vocab.tags))

    RNNsent_model(embed,rnn,projection,dropout,vocab)

end

function calc_scores(rm::RNNsent_model, data; average=true)
    B, Tx = size(data)
    
    emb = rm.embed(data)
    
    y = sum(rm.rnn(emb), dims=3) # nature of nll allows us to sum along each sentence

    return rm.projection(reshape(y,:,B))    

end

function loss_f(model, batch)  
    verify = deepcopy(batch[:,1]) # only tags allowed to in
    #mask!(verify,vocab.eos) no need to mask more :)
        
    scores = calc_scores(model,batch[:,2:end]) # trim one end
   
    return nll(scores,verify)/size(verify,1) # Loss for each sentence

end

function maploss(lossfn, model, data)
    total_loss = 0.0
    for part in collect(data)
        total_loss += lossfn(model,part)
    end

    return total_loss
end

model = RNNsent_model(512, 512, vocab; bidirectional=true, dropout=0.2)
rm = model

train_batches = collect(LMData(train))
test_batches = collect(LMData(test))
train_batches50 = train_batches[1:50] # Small sample for quick loss calculation

epoch = adam(loss_f, ((model, batch) for batch in train_batches))
bestmodel, bestloss = deepcopy(model), maploss(loss_f, model, test_batches)

progress!(ncycle(epoch, 100), seconds=5) do x
    global bestmodel, bestloss
    ## Report gradient norm for the first batch
    f = @diff loss_f(model,train_batches[1])
    gnorm = sqrt(sum(norm(grad(f,x))^2 for x in params(model)))
    ## Report training and validation loss
    trnloss = maploss(loss_f,model, train_batches50)
    devloss = maploss(loss_f,model, test_batches)
    ## Save model that does best on validation data
    if devloss < bestloss
        bestmodel, bestloss = deepcopy(model), devloss
    end
    (trn=exp(trnloss), dev=exp(devloss), ∇=gnorm)
end




