import os
import json
import time
import re
import mimetypes
import struct
from pathlib import Path
from openai import OpenAI
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# ---------------------------
# CONFIGURATION
# ---------------------------

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "Please set OPENAI_API_KEY environment variable. "
        "Run: export OPENAI_API_KEY='your-key-here'"
    )

if not GEMINI_API_KEY:
    raise ValueError(
        "Please set GEMINI_API_KEY environment variable. "
        "Run: export GEMINI_API_KEY='your-key-here'"
    )

# Output directory matches proposed structure
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "minimal_pairs"
NUM_PAIRS = 100   # change this to generate more
VOICE = "Zephyr"   # Gemini TTS voice options: Zephyr, Charon, Fenrir, Kore, Puck
RATE_LIMIT_SLEEP = 7  # seconds between API calls (TIER 1 allows ~10/min, so 7s is safe)
MAX_RETRIES = 5  # Maximum retries for rate limit errors

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------
# GPT-4o: Generate minimal pair sentence (with variety)
# ---------------------------

# Enhanced prompt with more variety
GEN_PROMPT_BASE = """
Generate a simple transitive sentence in English with a clear subject, verb, and object.
Avoid pronouns. Output only the sentence, no quotes.

Requirements:
- Use different subjects (names, animals, objects, professions)
- Use different verbs (ate, bought, found, saw, made, built, wrote, painted, etc.)
- Use different objects (food items, tools, books, vehicles, etc.)
- Make each sentence unique and different from previous ones

Example: "John ate the apple."
"""

def generate_sentence(pair_index, previous_sentences=None, max_previous=5):
    """Generate a simple sentence using GPT-4o with increased variety.
    
    Args:
        pair_index: Current pair index
        previous_sentences: List of previously generated sentences
        max_previous: Maximum number of previous sentences to include in prompt
    """
    try:
        prompt = GEN_PROMPT_BASE
        
        # Include recent sentences to encourage variety (limit to avoid token bloat)
        if previous_sentences:
            recent_sentences = previous_sentences[-max_previous:]  # Last N sentences
            if recent_sentences:
                prompt += "\n\nPreviously generated sentences (avoid similar ones):\n"
                for i, prev_sent in enumerate(recent_sentences, 1):
                    prompt += f"{i}. {prev_sent}\n"
        
        prompt += f"\nNow generate a NEW and DIFFERENT sentence (sentence #{pair_index + 1}):"
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9  # Increased temperature for more variety
        )
        sentence = completion.choices[0].message.content.strip()
        # Remove quotes if present
        sentence = sentence.strip('"').strip("'")
        # Remove any leading/trailing whitespace
        sentence = sentence.strip()
        return sentence
    except Exception as e:
        print(f"Error generating sentence: {e}")
        raise

# ---------------------------
# Gemini TTS generation (Subject vs Object Focus)
# ---------------------------

def add_emphasis(sentence, focus_word):
    """Adds Gemini emphasis tag around the chosen focus word."""
    # Replace the focus word with emphasized version
    # Handle case-insensitive replacement
    words = sentence.split()
    emphasized_words = []
    for word in words:
        # Remove punctuation for comparison
        word_clean = word.rstrip('.,!?;:')
        if word_clean.lower() == focus_word.lower():
            # Add emphasis with uppercase word and preserve punctuation
            punctuation = word[len(word_clean):]
            emphasized_words.append(f"<emphasis level='strong'>{word_clean.upper()}</emphasis>{punctuation}")
        else:
            emphasized_words.append(word)
    
    emphasized_sentence = " ".join(emphasized_words)
    return emphasized_sentence

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string."""
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}

def extract_retry_delay(error_message):
    """Extract retry delay from Gemini API error message."""
    # Look for "Please retry in X.XXXXs" pattern
    match = re.search(r'Please retry in ([\d.]+)s', str(error_message))
    if match:
        return float(match.group(1))
    return None

def synthesize_audio(text_with_emphasis, outfile):
    """Converts text with emphasis tags to WAV using Gemini TTS with rate limit handling."""
    # Ensure parent directory exists
    outfile.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare the content with instruction
    full_text = f"Read neutral except for emphasis word tag.\n{text_with_emphasis}"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=full_text),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=VOICE
                )
            )
        ),
    )
    
    # Retry logic for rate limits
    for attempt in range(MAX_RETRIES):
        try:
            # Collect audio chunks
            audio_chunks = []
            for chunk in gemini_client.models.generate_content_stream(
                model="gemini-2.5-flash-preview-tts",
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                if (chunk.candidates[0].content.parts[0].inline_data 
                    and chunk.candidates[0].content.parts[0].inline_data.data):
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    audio_chunks.append((inline_data.data, inline_data.mime_type))
            
            if not audio_chunks:
                raise ValueError("No audio data received from Gemini TTS")
            
            # Combine all audio chunks
            combined_audio = b"".join([chunk[0] for chunk in audio_chunks])
            mime_type = audio_chunks[0][1]  # Use first chunk's mime type
            
            # Convert to WAV if needed
            file_extension = mimetypes.guess_extension(mime_type)
            if file_extension is None or file_extension != ".wav":
                combined_audio = convert_to_wav(combined_audio, mime_type)
            
            # Save to file
            with open(outfile, "wb") as f:
                f.write(combined_audio)
            
            # Success!
            return
            
        except ClientError as e:
            # Check if it's a rate limit error (429)
            if hasattr(e, 'status_code') and e.status_code == 429:
                retry_delay = extract_retry_delay(str(e))
                if retry_delay:
                    wait_time = retry_delay + 1  # Add 1 second buffer
                    print(f"  Rate limit hit. Waiting {wait_time:.1f} seconds before retry (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Fallback: wait based on tier (TIER 1 = ~6 seconds between requests)
                    wait_time = 7
                    print(f"  Rate limit hit. Waiting {wait_time} seconds before retry (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                    continue
            else:
                # Not a rate limit error, re-raise
                print(f"Error synthesizing audio to {outfile}: {e}")
                raise
        except Exception as e:
            # Other errors, re-raise immediately
            print(f"Error synthesizing audio to {outfile}: {e}")
            raise
    
    # If we've exhausted all retries
    raise Exception(f"Failed to synthesize audio after {MAX_RETRIES} retries due to rate limits")

# ---------------------------
# Utility: Extract subject & object (LLM)
# ---------------------------

EXTRACTION_PROMPT = """
Given this sentence, extract:
1. subject (single word)
2. object (single word)

Return ONLY valid JSON in this format: {"subject": "...", "object": "..."}

Sentence:
"""

def extract_subject_object(sentence):
    """Extract subject and object from sentence using GPT-4o."""
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + sentence}],
            temperature=0,
            response_format={"type": "json_object"}  # Force JSON response
        )
        result = json.loads(completion.choices[0].message.content)
        return result
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise
    except Exception as e:
        print(f"Error extracting subject/object: {e}")
        raise

# ---------------------------
# MAIN PIPELINE
# ---------------------------

def find_existing_pairs():
    """Find all existing pair directories and return their indices."""
    existing_pairs = set()
    if not OUTPUT_DIR.exists():
        return existing_pairs
    
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir() and item.name.startswith("pair_"):
            try:
                # Extract the number from "pair_0000"
                pair_num = int(item.name.split("_")[1])
                # Check if it's a complete pair (has text.txt and labels.json)
                if (item / "text.txt").exists() and (item / "labels.json").exists():
                    existing_pairs.add(pair_num)
            except (ValueError, IndexError):
                # Skip directories that don't match the pattern
                continue
    
    return existing_pairs

def load_existing_sentences():
    """Load sentences from existing pairs to maintain variety."""
    sentences = []
    existing_pairs = find_existing_pairs()
    
    for pair_num in sorted(existing_pairs):
        pair_dir = OUTPUT_DIR / f"pair_{pair_num:04d}"
        text_file = pair_dir / "text.txt"
        if text_file.exists():
            try:
                sentence = text_file.read_text().strip()
                if sentence:
                    sentences.append(sentence)
            except Exception:
                continue
    
    return sentences

def get_next_available_pair_index(existing_pairs, start_index=0):
    """Find the next available pair index, skipping existing ones."""
    for i in range(start_index, start_index + NUM_PAIRS * 2):  # Search up to 2x NUM_PAIRS to find gaps
        if i not in existing_pairs:
            return i
    # If no gap found, return the next sequential index
    if existing_pairs:
        return max(existing_pairs) + 1
    return 0

def main():
    """Main pipeline to generate minimal pairs dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for existing pairs
    existing_pairs = find_existing_pairs()
    existing_sentences = load_existing_sentences()
    
    successful_pairs = 0
    failed_pairs = 0
    generated_sentences = existing_sentences.copy()  # Start with existing sentences for variety
    generated_sentences_set = set(generated_sentences)  # Track sentences as set for fast duplicate checking
    
    print(f"Starting generation of {NUM_PAIRS} minimal pairs...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using Gemini TTS voice: {VOICE}")
    if existing_pairs:
        print(f"Found {len(existing_pairs)} existing pairs: {sorted(existing_pairs)}")
        print(f"Loaded {len(existing_sentences)} existing sentences for variety")
    print("-" * 60)
    
    # Find starting index
    current_index = get_next_available_pair_index(existing_pairs, 0)
    pairs_generated = 0
    
    while pairs_generated < NUM_PAIRS:
        # Skip if this pair already exists
        if current_index in existing_pairs:
            print(f"\nSkipping pair_{current_index:04d} (already exists)")
            current_index += 1
            continue
        
        try:
            print(f"\n[{pairs_generated + 1}/{NUM_PAIRS}] Generating pair_{current_index:04d}…")
            
            # Step 1: Generate new minimal pair sentence (with retry for uniqueness)
            max_retries = 5
            sentence = None
            for retry in range(max_retries):
                candidate = generate_sentence(current_index, previous_sentences=generated_sentences)
                if candidate not in generated_sentences_set:
                    sentence = candidate
                    generated_sentences.append(candidate)
                    generated_sentences_set.add(candidate)
                    break
                print(f"  Retry {retry + 1}: Duplicate sentence detected, generating new one...")
            
            if sentence is None:
                print(f"  Warning: Could not generate unique sentence after {max_retries} retries. Skipping...")
                failed_pairs += 1
                current_index += 1
                continue
            
            print(f"  Generated sentence: {sentence}")
            
            # Step 2: Extract subject + object
            roles = extract_subject_object(sentence)
            subject = roles["subject"]
            obj = roles["object"]
            print(f"  Subject: {subject}, Object: {obj}")
            
            # Validate that subject and object are in the sentence
            sentence_words = [w.rstrip('.,!?;:').lower() for w in sentence.split()]
            if subject.lower() not in sentence_words or obj.lower() not in sentence_words:
                print(f"  Warning: Subject or object not found in sentence. Skipping...")
                failed_pairs += 1
                current_index += 1
                continue
            
            # Create pair directory
            pair_dir = OUTPUT_DIR / f"pair_{current_index:04d}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            # Save text file
            (pair_dir / "text.txt").write_text(sentence)
            
            # Step 3: Create text with emphasis tags (Gemini format, not SSML)
            text_subject = add_emphasis(sentence, subject)
            text_object = add_emphasis(sentence, obj)
            
            # Step 4: Generate audio files using Gemini TTS
            print(f"  Generating audio with subject focus...")
            synthesize_audio(text_subject, pair_dir / "audio_focus_subject.wav")
            time.sleep(RATE_LIMIT_SLEEP)
            
            print(f"  Generating audio with object focus...")
            synthesize_audio(text_object, pair_dir / "audio_focus_object.wav")
            time.sleep(RATE_LIMIT_SLEEP)
            
            # Step 5: Write JSON labels
            tokens = [w.rstrip('.,!?;:') for w in sentence.split()]
            try:
                subject_index = tokens.index(subject)
            except ValueError:
                # Try case-insensitive
                subject_index = next(i for i, w in enumerate(tokens) if w.lower() == subject.lower())
            
            try:
                obj_index = tokens.index(obj)
            except ValueError:
                # Try case-insensitive
                obj_index = next(i for i, w in enumerate(tokens) if w.lower() == obj.lower())
            
            labels = {
                "sentence": sentence,
                "tokens": tokens,
                "audio_files": {
                    "audio_focus_subject.wav": {
                        "focus_token": subject,
                        "focus_index": subject_index,
                        "type": "contrastive"
                    },
                    "audio_focus_object.wav": {
                        "focus_token": obj,
                        "focus_index": obj_index,
                        "type": "contrastive"
                    }
                }
            }
            
            with open(pair_dir / "labels.json", "w") as f:
                json.dump(labels, f, indent=2)
            
            successful_pairs += 1
            pairs_generated += 1
            current_index += 1
            print(f"  ✓ Generated {pair_dir.name}")
            
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user.")
            break
        except Exception as e:
            print(f"  ✗ Error generating pair_{current_index:04d}: {e}")
            import traceback
            traceback.print_exc()
            failed_pairs += 1
            current_index += 1
            # Continue to next pair instead of crashing
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Successful pairs: {successful_pairs}")
    print(f"Failed pairs: {failed_pairs}")
    print(f"Total attempted: {successful_pairs + failed_pairs}")
    print(f"Unique sentences: {len(generated_sentences)}")
    print(f"\nDataset ready at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
