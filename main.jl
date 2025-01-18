using HTTP
using DelimitedFiles
using Statistics
using Base.Threads
using StatsBase
using Unicode
using Printf

include("stopwords.jl")

# Constants
const BASE_URL = "https://gutenberg.net.au/ebooks"
global total_tokenization_time = 0.0
global total_processed_documents = 0
# Function to handle invalid characters
function handle_invalid_char(c)
    try
        return isvalid(c) ? lowercase(c) : ' '
    catch e
        if isa(e, Base.InvalidCharError)
            return ' '
        else
            rethrow(e)
        end
    end
end

# Function to count words in a document
function count_words(url, frequencies, mutex)
    txt = ""
    try
        response = HTTP.get(url)
        txt = String(response.body)
    catch e
        if isa(e, HTTP.Exceptions.StatusError) && e.status == 404
            println("Document not found at URL: $url")
        else
            println("Error accessing URL $url: $e")
        end
        return
    end

    # Start measuring tokenization time
    start_tokenization = time()

    cleaned_txt = map(handle_invalid_char, txt)
    words = [m.match for m in eachmatch(r"\b\w+\b", cleaned_txt)]
    word_count = length(words)

    # Finish measuring tokenization time
    finish_tokenization = time()

    # Calculate time taken for tokenization
    tokenization_time = finish_tokenization - start_tokenization

    # Determine the number of words processed
    total_words = word_count

    # Calculate word tokenization speed
    tokenization_speed = total_words / tokenization_time

    # Print word tokenization speed / Velocidad de tokenización de palabras 
    println("\nWord Tokenization Speed: $tokenization_speed words per second")

    # Lock the mutex to safely update shared variables
    lock(mutex) do
        # Update total word count and total tokenization time
        global total_tokenization_time += tokenization_time
        global total_processed_documents += 1
    end

    #doc_number = parse(Int, match(r"\d+", url).match)
    doc_number_match = match(r"/(\d+)\.txt$", url)
    doc_number = doc_number_match === nothing ? 0 : parse(Int, doc_number_match.captures[1])

    lock(mutex) do
        frequencies[url] = Dict("words" => StatsBase.countmap(words), "total_words" => word_count, "doc_number" => doc_number)
        global finished_count += 1

        println("Processed document $finished_count - Doc number $doc_number: $url")
    end
end


# Function to calculate IDF
function calculate_idf(documents, term)
    document_count = length(documents)
    term_occurrences = sum(1 for document in documents if get(document["words"], term, 0) > 0; init=0)
    return log(document_count / (term_occurrences + 1)) + 1
end

# Main function
function main()
    # Initialize data structures
    frequencies = Dict()
    mutex = ReentrantLock()
    global finished_count = 0

    # Processing documents in parallel
    start = time()
    threads = []
  
    
    for i in 1:50
        folder_number = div(i - 1, 100) + 1
        folder = lpad(folder_number, 2, '0')
        
        # Reset file number for each new folder
        file_pattern = ((i - 1) % 100) * 10 + 11
        file_number = lpad(file_pattern, 5, '0') # Change the padding to 5
        
        url = "http://gutenberg.net.au/ebooks$(folder)/$(folder)$(file_number).txt"
        push!(threads, Threads.@spawn count_words(url, frequencies, mutex))
    end
    for thread in threads
        wait(thread)
    end

    # Calculate and print the average tokenization speed
    average_tokenization_speed = total_tokenization_time / total_processed_documents
    println("\nAverage Word Tokenization Speed: $average_tokenization_speed words per second")
    
    finish = time()

    # Determine the number of documents processed
    total_documents = length(threads)
    # Calculate throughput/ rendimiento 
    throughput = total_documents / (finish - start)

    # Print the throughput
    println("\nThroughput: $throughput documents per second")

    println("Total Processed Documents: $total_documents")

    stop_words = StopWords.get_stop_words() #stopword.jl

    # Calculate frequency of words across all documents
    all_words = Dict{String, Int}()
    for (url, data) in pairs(frequencies)
        for (word, count) in pairs(data["words"])
            if !(word in stop_words)
                all_words[word] = get(all_words, word, 0) + count
            end
        end
    end

    # Find the 10 most frequent words
    sorted_words = sort(collect(all_words), by = tuple -> last(tuple), rev=true)
    most_frequent_words = sorted_words[1:min(20, length(sorted_words))]

    println("\nThe 10 most frequent words are:")
    for (word, count) in most_frequent_words
        println("$word: $count")
    end

    # Calculate TF-IDF
    terms = ["eyes", "house", "room", "hand", "night"]
    tfidf_scores = Dict(term => [] for term in terms)

    for term in terms
        for (url, data) in pairs(frequencies)
            tf = get(data["words"], term, 0) / data["total_words"]
            idf = calculate_idf(values(frequencies), term)
            tfidf = tf * idf
            doc_number = data["doc_number"]
            push!(tfidf_scores[term], Dict("url" => url, "score" => tfidf, "document_number" => doc_number))
        end
    end

    # Sort and print TF-IDF scores
    sorted_tfidf_scores = Dict(term => sort(scores, rev=true, by=x -> x["score"]) for (term, scores) in pairs(tfidf_scores))

    println("\nTF-IDF Scores:")
    for (term, scores) in pairs(sorted_tfidf_scores)
        println("\nTerm: $term")
        for score in scores
            println("  Document: $(score["document_number"]), Score: $(score["score"])")
        end
    end
end 
# Run main if this file is the main program
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end