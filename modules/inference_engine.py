"""
Inference Engine Module for HF-LLM-RUNNER
Handles text generation with memory-efficient inference
"""

import gc
import time
from typing import Dict, Optional, Callable, Any, Generator, Union
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.memory_tracker import MemoryTracker

logger = setup_logger(__name__)


class InferenceEngine:
    """
    Inference engine for text generation with memory management.
    
    Features:
    - Streaming output for reduced memory footprint
    - Progress callbacks for long generations
    - Memory cleanup between generations
    - KV-cache management
    
    Usage:
        engine = InferenceEngine(model, tokenizer)
        output = engine.generate("Hello", max_new_tokens=100)
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        deepspeed_config: Optional[Dict] = None,
        cleanup_interval: int = 10
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Loaded model (from ModelLoader)
            tokenizer: Loaded tokenizer
            deepspeed_config: Optional DeepSpeed configuration
            cleanup_interval: Tokens between memory cleanup (0 to disable)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.deepspeed_config = deepspeed_config
        self.cleanup_interval = cleanup_interval
        
        self.memory_tracker = MemoryTracker()
        self._is_deepspeed = deepspeed_config is not None
    
    def _check_cuda(self):
        """Check CUDA availability"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _cleanup_memory(self):
        """Perform memory cleanup"""
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
    
    def _get_device(self):
        """Get model device"""
        try:
            import torch
            if hasattr(self.model, 'device'):
                return self.model.device
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            return 'cpu'
    
    def encode_input(self, prompt: str) -> Any:
        """
        Encode input prompt to tensor.
        
        Args:
            prompt: Input text
            
        Returns:
            Input tensor
        """
        device = self._get_device()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # Safety limit
        )
        
        # Move to device
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        else:
            import torch
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
    
    def decode_output(self, output_ids: Any, skip_special_tokens: bool = True) -> str:
        """
        Decode output tokens to text.
        
        Args:
            output_ids: Output token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            output_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling (vs greedy)
            stream: Whether to stream output (generator mode)
            **kwargs: Additional generation kwargs
            
        Returns:
            Generated text (or generator if stream=True)
        """
        if stream:
            return self._generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                **kwargs
            )
        
        return self._generate_batch(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            **kwargs
        )
    
    def _generate_batch(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        **kwargs
    ) -> str:
        """
        Generate text in batch mode (non-streaming).
        """
        logger.info(f"Generating {max_new_tokens} tokens...")
        logger.info(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}")
        
        # Record memory before generation
        self.memory_tracker.record_snapshot("Before generation")
        
        # Encode input
        inputs = self.encode_input(prompt)
        input_length = inputs['input_ids'].shape[1] if hasattr(inputs, 'keys') else inputs.shape[1]
        
        # Generate
        start_time = time.time()
        
        try:
            import torch
            
            # Build generation config
            generation_config = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'repetition_penalty': repetition_penalty,
                'do_sample': do_sample,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'use_cache': True,
            }
            
            # Add any additional kwargs
            generation_config.update(kwargs)
            
            # Handle DeepSpeed
            if self._is_deepspeed:
                # DeepSpeed has different generation API
                outputs = self._generate_deepspeed(
                    inputs=inputs,
                    generation_config=generation_config
                )
            else:
                # Standard HF generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
            
            # Decode output
            generated_ids = outputs[0][input_length:]
            generated_text = self.decode_output(generated_ids)
            
            # Calculate stats
            elapsed = time.time() - start_time
            tokens_generated = len(generated_ids)
            tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0
            
            logger.info(f"Generation completed in {elapsed:.2f}s")
            logger.info(f"Generated {tokens_generated} tokens ({tokens_per_second:.2f} tok/s)")
            
            # Record memory after generation
            self.memory_tracker.record_snapshot("After generation")
            
            # Cleanup
            if self.cleanup_interval > 0:
                self._cleanup_memory()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate text in streaming mode.
        
        Yields tokens as they are generated.
        """
        logger.info(f"Starting streaming generation for {max_new_tokens} tokens...")
        
        # Encode input
        inputs = self.encode_input(prompt)
        input_length = inputs['input_ids'].shape[1] if hasattr(inputs, 'keys') else inputs.shape[1]
        
        try:
            import torch
            from transformers import TextIteratorStreamer
            import threading
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Generation config
            generation_config = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'repetition_penalty': repetition_penalty,
                'do_sample': do_sample,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'streamer': streamer,
                'use_cache': True,
            }
            generation_config.update(kwargs)
            
            # Run generation in thread
            def generate_thread():
                with torch.no_grad():
                    self.model.generate(**inputs, **generation_config)
            
            thread = threading.Thread(target=generate_thread)
            thread.start()
            
            # Yield tokens as they come
            for token in streamer:
                yield token
                
                # Cleanup periodically
                if self.cleanup_interval > 0:
                    self._cleanup_memory()
            
            thread.join()
            
        except ImportError:
            # Fallback to batch if streaming not available
            logger.warning("Streaming not available, falling back to batch generation")
            result = self._generate_batch(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                **kwargs
            )
            yield result
    
    def _generate_deepspeed(self, inputs: Any, generation_config: Dict) -> Any:
        """
        Generate with DeepSpeed engine.
        
        DeepSpeed has a slightly different generation API.
        """
        try:
            import torch
            
            # DeepSpeed generate method
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
            
            return outputs
            
        except Exception as e:
            logger.error(f"DeepSpeed generation failed: {e}")
            raise
    
    def generate_with_progress(
        self,
        prompt: str,
        max_new_tokens: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        progress_interval: int = 10,
        **kwargs
    ) -> str:
        """
        Generate text with progress updates.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            progress_callback: Callback function(current, total, partial_text)
            progress_interval: Update interval in tokens
            **kwargs: Generation kwargs
            
        Returns:
            Generated text
        """
        logger.info(f"Generating with progress updates every {progress_interval} tokens")
        
        # Encode input
        inputs = self.encode_input(prompt)
        input_length = inputs['input_ids'].shape[1] if hasattr(inputs, 'keys') else inputs.shape[1]
        
        try:
            import torch
            
            # Track partial output
            partial_text = ""
            tokens_generated = 0
            
            # Use streaming internally
            for token in self._generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                **kwargs
            ):
                partial_text += token
                tokens_generated += 1
                
                # Call progress callback
                if progress_callback and tokens_generated % progress_interval == 0:
                    progress_callback(
                        tokens_generated,
                        max_new_tokens,
                        partial_text
                    )
            
            # Final callback
            if progress_callback:
                progress_callback(max_new_tokens, max_new_tokens, partial_text)
            
            return partial_text
            
        except Exception as e:
            logger.error(f"Generation with progress failed: {e}")
            raise
    
    def chat(
        self,
        messages: list,
        max_new_tokens: int = 500,
        **kwargs
    ) -> str:
        """
        Generate response for chat-style conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            **kwargs: Generation kwargs
            
        Returns:
            Model response
        """
        # Format messages using tokenizer's chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt += f"{role.capitalize()}: {content}\n"
            prompt += "Assistant:"
        
        return self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)


class LongTextGenerator(InferenceEngine):
    """
    Specialized generator for very long text outputs.
    
    Optimized for generating hundreds or thousands of lines
    like the user's poem example (500 lines, 15 words each, 50 chars per word).
    """
    
    def generate_long_text(
        self,
        prompt: str,
        target_lines: int = 500,
        words_per_line: int = 15,
        chars_per_word: int = 50,
        **kwargs
    ) -> str:
        """
        Generate very long text with optimized settings.
        
        Args:
            prompt: Input prompt
            target_lines: Target number of lines
            words_per_line: Words per line
            chars_per_word: Characters per word
            **kwargs: Generation kwargs
            
        Returns:
            Generated text
        """
        # Calculate approximate tokens needed
        # Rough estimate: 1 token ≈ 4 characters
        chars_per_line = words_per_line * chars_per_word
        total_chars = target_lines * chars_per_line
        estimated_tokens = total_chars // 4  # Rough estimate
        
        logger.info(f"Generating long text: ~{target_lines} lines, ~{estimated_tokens} tokens")
        
        # Use higher repetition penalty for long text
        kwargs.setdefault('repetition_penalty', 1.2)
        
        # Generate
        return self.generate(
            prompt=prompt,
            max_new_tokens=min(estimated_tokens, 4000),  # Cap at model limit
            stream=True,
            **kwargs
        )
    
    def generate_in_chunks(
        self,
        prompt: str,
        total_tokens: int,
        chunk_size: int = 500,
        overlap: int = 50,
        **kwargs
    ) -> str:
        """
        Generate very long text in chunks.
        
        This helps with memory management for extremely long outputs.
        
        Args:
            prompt: Initial prompt
            total_tokens: Total tokens to generate
            chunk_size: Tokens per chunk
            overlap: Tokens to overlap between chunks
            **kwargs: Generation kwargs
            
        Returns:
            Complete generated text
        """
        full_text = ""
        current_prompt = prompt
        tokens_generated = 0
        
        while tokens_generated < total_tokens:
            # Calculate tokens for this chunk
            remaining = total_tokens - tokens_generated
            chunk_tokens = min(chunk_size, remaining)
            
            logger.info(f"Generating chunk: {tokens_generated}/{total_tokens} tokens")
            
            # Generate chunk
            chunk_text = self.generate(
                prompt=current_prompt,
                max_new_tokens=chunk_tokens,
                **kwargs
            )
            
            full_text += chunk_text
            tokens_generated += chunk_tokens
            
            # Update prompt for next chunk (with overlap)
            if tokens_generated < total_tokens:
                # Use last 'overlap' tokens as context
                words = chunk_text.split()
                context = " ".join(words[-overlap:])
                current_prompt = f"{prompt}\n\nContinue from: {context}"
            
            # Memory cleanup between chunks
            self._cleanup_memory()
        
        return full_text


if __name__ == "__main__":
    print("InferenceEngine module loaded successfully")
