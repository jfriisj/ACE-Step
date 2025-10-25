"""
WAV File Diagnostic Tool for ACE-Step
Analyzes audio files for corruption, format issues, and quality problems
"""

import os
import sys
import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

class WAVDiagnostic:
    """Comprehensive WAV file analysis tool"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.results = {
            "file_exists": False,
            "file_size": 0,
            "is_valid_wav": False,
            "format_info": {},
            "audio_stats": {},
            "issues": [],
            "recommendations": []
        }
    
    def analyze(self):
        """Perform comprehensive analysis of the WAV file"""
        print(f"🔍 Analyzing WAV file: {self.file_path}")
        print("=" * 50)
        
        # Basic file checks
        self._check_file_exists()
        if not self.results["file_exists"]:
            return self.results
        
        # File format analysis
        self._analyze_wav_format()
        self._check_wav_header()
        self._analyze_audio_content()
        self._check_for_corruption()
        self._generate_recommendations()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _check_file_exists(self):
        """Check if file exists and get basic info"""
        if self.file_path.exists():
            self.results["file_exists"] = True
            self.results["file_size"] = self.file_path.stat().st_size
            print(f"✅ File exists: {self.file_path}")
            print(f"📁 File size: {self.results['file_size']:,} bytes ({self.results['file_size']/1024/1024:.2f} MB)")
        else:
            self.results["issues"].append("File does not exist")
            print(f"❌ File not found: {self.file_path}")
    
    def _analyze_wav_format(self):
        """Analyze WAV file format using wave module"""
        try:
            with wave.open(str(self.file_path), 'rb') as wav_file:
                self.results["is_valid_wav"] = True
                
                # Get format information
                format_info = {
                    "channels": wav_file.getnchannels(),
                    "sample_width": wav_file.getsampwidth(),
                    "sample_rate": wav_file.getframerate(),
                    "frame_count": wav_file.getnframes(),
                    "duration": wav_file.getnframes() / wav_file.getframerate()
                }
                
                self.results["format_info"] = format_info
                
                print(f"✅ Valid WAV format")
                print(f"🎵 Channels: {format_info['channels']} ({'Stereo' if format_info['channels'] == 2 else 'Mono' if format_info['channels'] == 1 else 'Multi-channel'})")
                print(f"📊 Sample Rate: {format_info['sample_rate']:,} Hz")
                print(f"🔢 Sample Width: {format_info['sample_width']} bytes ({format_info['sample_width'] * 8}-bit)")
                print(f"⏱️  Duration: {format_info['duration']:.2f} seconds")
                print(f"📈 Frame Count: {format_info['frame_count']:,}")
                
        except wave.Error as e:
            self.results["issues"].append(f"WAV format error: {str(e)}")
            print(f"❌ WAV format error: {e}")
        except Exception as e:
            self.results["issues"].append(f"File read error: {str(e)}")
            print(f"❌ File read error: {e}")
    
    def _check_wav_header(self):
        """Manually check WAV file header for corruption"""
        try:
            with open(self.file_path, 'rb') as f:
                # Read first 44 bytes (standard WAV header)
                header = f.read(44)
                
                if len(header) < 44:
                    self.results["issues"].append("Incomplete WAV header (file too small)")
                    print("❌ Incomplete WAV header")
                    return
                
                # Check RIFF signature
                if header[:4] != b'RIFF':
                    self.results["issues"].append("Missing RIFF signature")
                    print("❌ Invalid RIFF signature")
                
                # Check WAV signature
                if header[8:12] != b'WAVE':
                    self.results["issues"].append("Missing WAVE signature")
                    print("❌ Invalid WAVE signature")
                
                # Check fmt chunk
                if header[12:16] != b'fmt ':
                    self.results["issues"].append("Missing fmt chunk")
                    print("❌ Invalid fmt chunk")
                
                # Extract format details
                fmt_size = struct.unpack('<I', header[16:20])[0]
                audio_format = struct.unpack('<H', header[20:22])[0]
                
                if audio_format != 1:
                    self.results["issues"].append(f"Non-PCM audio format: {audio_format}")
                    print(f"⚠️  Non-PCM format detected: {audio_format}")
                
                print("✅ WAV header structure is valid")
                
        except Exception as e:
            self.results["issues"].append(f"Header analysis error: {str(e)}")
            print(f"❌ Header analysis error: {e}")
    
    def _analyze_audio_content(self):
        """Analyze the actual audio data"""
        if not self.results["is_valid_wav"]:
            return
        
        try:
            # Read audio data
            with wave.open(str(self.file_path), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                
            # Convert to numpy array
            if self.results["format_info"]["sample_width"] == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
            elif self.results["format_info"]["sample_width"] == 4:
                audio_data = np.frombuffer(frames, dtype=np.int32)
            else:
                print(f"⚠️  Unsupported sample width: {self.results['format_info']['sample_width']}")
                return
            
            # Reshape for stereo
            if self.results["format_info"]["channels"] == 2:
                audio_data = audio_data.reshape(-1, 2)
                left_channel = audio_data[:, 0]
                right_channel = audio_data[:, 1]
                
                # Analyze both channels
                self._analyze_channel(left_channel, "Left")
                self._analyze_channel(right_channel, "Right")
                
                # Check for stereo issues
                if np.array_equal(left_channel, right_channel):
                    self.results["issues"].append("Identical stereo channels (fake stereo)")
                    print("⚠️  Identical stereo channels detected")
                
            else:
                self._analyze_channel(audio_data, "Mono")
            
            # Overall statistics
            audio_stats = {
                "max_amplitude": float(np.max(np.abs(audio_data))),
                "rms_level": float(np.sqrt(np.mean(audio_data.astype(np.float64)**2))),
                "dynamic_range": float(np.max(audio_data) - np.min(audio_data)),
                "zero_crossings": int(np.sum(np.diff(np.sign(audio_data.flatten())) != 0)),
                "silence_ratio": float(np.sum(audio_data == 0) / len(audio_data.flatten()))
            }
            
            self.results["audio_stats"] = audio_stats
            
            print(f"📊 Audio Statistics:")
            print(f"   Max Amplitude: {audio_stats['max_amplitude']:,.0f}")
            print(f"   RMS Level: {audio_stats['rms_level']:,.0f}")
            print(f"   Dynamic Range: {audio_stats['dynamic_range']:,.0f}")
            print(f"   Zero Crossings: {audio_stats['zero_crossings']:,}")
            print(f"   Silence Ratio: {audio_stats['silence_ratio']:.3%}")
            
        except Exception as e:
            self.results["issues"].append(f"Audio analysis error: {str(e)}")
            print(f"❌ Audio analysis error: {e}")
    
    def _analyze_channel(self, channel_data, channel_name):
        """Analyze individual audio channel"""
        # Check for silence
        if np.all(channel_data == 0):
            self.results["issues"].append(f"{channel_name} channel is completely silent")
            print(f"❌ {channel_name} channel is silent")
            return
        
        # Check for clipping
        max_val = np.iinfo(channel_data.dtype).max
        min_val = np.iinfo(channel_data.dtype).min
        
        clipped_samples = np.sum((channel_data == max_val) | (channel_data == min_val))
        if clipped_samples > 0:
            clipping_ratio = clipped_samples / len(channel_data)
            self.results["issues"].append(f"{channel_name} channel has {clipping_ratio:.3%} clipped samples")
            print(f"⚠️  {channel_name} channel clipping: {clipping_ratio:.3%}")
        
        # Check for DC offset
        dc_offset = np.mean(channel_data.astype(np.float64))
        if abs(dc_offset) > 100:  # Threshold for significant DC offset
            self.results["issues"].append(f"{channel_name} channel has DC offset: {dc_offset:.1f}")
            print(f"⚠️  {channel_name} channel DC offset: {dc_offset:.1f}")
        
        print(f"✅ {channel_name} channel analysis complete")
    
    def _check_for_corruption(self):
        """Check for signs of file corruption"""
        # Check if file size matches expected size
        if self.results["is_valid_wav"] and self.results["format_info"]:
            expected_data_size = (
                self.results["format_info"]["frame_count"] * 
                self.results["format_info"]["channels"] * 
                self.results["format_info"]["sample_width"]
            )
            header_size = 44  # Standard WAV header size
            expected_file_size = expected_data_size + header_size
            
            size_difference = abs(self.results["file_size"] - expected_file_size)
            
            if size_difference > 1000:  # Allow small differences
                self.results["issues"].append(f"File size mismatch: expected ~{expected_file_size:,}, got {self.results['file_size']:,}")
                print(f"⚠️  File size mismatch: {size_difference:,} bytes difference")
            else:
                print("✅ File size matches expected size")
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if not self.results["issues"]:
            recommendations.append("✅ No issues detected - file appears to be healthy")
        
        # Format recommendations
        if self.results["is_valid_wav"] and self.results["format_info"]:
            sr = self.results["format_info"]["sample_rate"]
            if sr < 44100:
                recommendations.append(f"Consider using higher sample rate (current: {sr} Hz)")
            
            if self.results["format_info"]["sample_width"] == 1:
                recommendations.append("Consider using higher bit depth (current: 8-bit)")
        
        # Audio quality recommendations
        if self.results["audio_stats"]:
            if self.results["audio_stats"]["silence_ratio"] > 0.5:
                recommendations.append("High silence ratio detected - check if audio generation completed properly")
            
            if self.results["audio_stats"]["max_amplitude"] < 1000:
                recommendations.append("Low audio level detected - consider normalizing")
        
        # General recommendations
        if "clipped" in str(self.results["issues"]):
            recommendations.append("Reduce audio levels to prevent clipping")
        
        if "DC offset" in str(self.results["issues"]):
            recommendations.append("Apply DC offset removal filter")
        
        self.results["recommendations"] = recommendations
    
    def _print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 50)
        print("📋 ANALYSIS SUMMARY")
        print("=" * 50)
        
        if not self.results["issues"]:
            print("🎉 No issues detected! File appears to be healthy.")
        else:
            print(f"⚠️  {len(self.results['issues'])} issue(s) detected:")
            for i, issue in enumerate(self.results["issues"], 1):
                print(f"   {i}. {issue}")
        
        if self.results["recommendations"]:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(self.results["recommendations"], 1):
                print(f"   {i}. {rec}")
        
        print("=" * 50)
    
    def save_report(self, output_file: str = None):
        """Save detailed analysis report"""
        if output_file is None:
            output_file = f"{self.file_path.stem}_diagnostic_report.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Detailed report saved to: {output_file}")
    
    def create_waveform_plot(self, output_file: str = None):
        """Create waveform visualization"""
        if not self.results["is_valid_wav"]:
            print("❌ Cannot create waveform - invalid WAV file")
            return
        
        try:
            # Read audio data for plotting
            with wave.open(str(self.file_path), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
            
            # Convert to numpy array
            if self.results["format_info"]["sample_width"] == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
            else:
                audio_data = np.frombuffer(frames, dtype=np.int32)
            
            # Time axis
            duration = len(audio_data) / sample_rate / channels
            time_axis = np.linspace(0, duration, len(audio_data) // channels)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2)
                plt.subplot(2, 1, 1)
                plt.plot(time_axis, audio_data[:, 0])
                plt.title("Left Channel")
                plt.ylabel("Amplitude")
                
                plt.subplot(2, 1, 2)
                plt.plot(time_axis, audio_data[:, 1])
                plt.title("Right Channel")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Amplitude")
            else:
                plt.plot(time_axis, audio_data)
                plt.title("Waveform")
                plt.xlabel("Time (seconds)")
                plt.ylabel("Amplitude")
            
            plt.tight_layout()
            
            if output_file is None:
                output_file = f"{self.file_path.stem}_waveform.png"
            
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📈 Waveform plot saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Error creating waveform plot: {e}")

def main():
    """Main diagnostic function"""
    if len(sys.argv) < 2:
        print("Usage: python wav_diagnostic.py <wav_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Create diagnostic instance
    diagnostic = WAVDiagnostic(file_path)
    
    # Run analysis
    results = diagnostic.analyze()
    
    # Save detailed report
    diagnostic.save_report()
    
    # Create waveform plot
    diagnostic.create_waveform_plot()
    
    return results

if __name__ == "__main__":
    main()