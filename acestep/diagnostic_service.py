"""
ACE-Step Diagnostic Service

Integrates WAV file diagnostics into the generation pipeline for automated quality assurance.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Import the diagnostic tool
from .wav_diagnostic import WAVDiagnostic

logger = logging.getLogger(__name__)

class DiagnosticService:
    """Service to automatically diagnose generated audio files"""
    
    def __init__(self, diagnostic_dir: Optional[str] = None, auto_diagnose: bool = True):
        """
        Initialize diagnostic service
        
        Args:
            diagnostic_dir: Directory to save diagnostic reports (defaults to outputs/diagnostics)
            auto_diagnose: Whether to automatically run diagnostics on saved files
        """
        # Use environment variables for configuration
        self.auto_diagnose = auto_diagnose if auto_diagnose is not None else os.environ.get("ACE_AUTO_DIAGNOSE", "true").lower() == "true"
        self.diagnostic_dir = diagnostic_dir or os.environ.get("ACE_DIAGNOSTIC_DIR") or os.path.join(os.environ.get("ACE_OUTPUT_DIR", "./outputs"), "diagnostics")
        
        # Additional configuration from environment
        self.enable_plots = os.environ.get("ACE_DIAGNOSTIC_PLOTS", "false").lower() == "true"
        self.max_file_size_mb = int(os.environ.get("ACE_DIAGNOSTIC_MAX_SIZE", "100"))
        self.cleanup_days = int(os.environ.get("ACE_DIAGNOSTIC_CLEANUP_DAYS", "30"))
        
        # Ensure diagnostic directory exists
        os.makedirs(self.diagnostic_dir, exist_ok=True)
        
        logger.info(f"Diagnostic service initialized - Auto-diagnose: {auto_diagnose}, Reports dir: {self.diagnostic_dir}")
    
    def diagnose_file(self, file_path: str, save_report: bool = True) -> Dict[str, Any]:
        """
        Run diagnostics on a generated audio file
        
        Args:
            file_path: Path to the audio file to diagnose
            save_report: Whether to save the diagnostic report to file
            
        Returns:
            Dictionary containing diagnostic results
        """
        try:
            # Check file size limits
            if self.max_file_size_mb > 0:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb > self.max_file_size_mb:
                    logger.warning(f"Skipping large file ({file_size_mb:.1f}MB > {self.max_file_size_mb}MB): {file_path}")
                    return {
                        'file_path': file_path,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'errors': [f"File too large for analysis ({file_size_mb:.1f}MB > {self.max_file_size_mb}MB)"],
                        'warnings': [],
                        'analysis': {'file_size_mb': file_size_mb},
                        'success': False
                    }
            
            logger.info(f"Running diagnostics on: {file_path}")
            
            # Create diagnostic tool instance for this file
            wav_diagnostic = WAVDiagnostic(file_path)
            
            # Run the diagnostic analysis
            diagnostic_results = wav_diagnostic.analyze()
            
            # Convert to our standard format
            results = {
                'file_path': file_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'errors': [],
                'warnings': [],
                'analysis': {},
                'success': diagnostic_results.get('is_valid_wav', False),
                'raw_results': diagnostic_results
            }
            
            # Extract issues as errors/warnings
            issues = diagnostic_results.get('issues', [])
            for issue in issues:
                if any(keyword in issue.lower() for keyword in ['error', 'corrupt', 'invalid', 'missing']):
                    results['errors'].append(issue)
                else:
                    results['warnings'].append(issue)
            
            # Extract analysis data
            if 'format_info' in diagnostic_results:
                format_info = diagnostic_results['format_info']
                results['analysis'].update({
                    'sample_rate': format_info.get('sample_rate'),
                    'channels': format_info.get('channels'),
                    'bit_depth': format_info.get('sample_width', 0) * 8,
                    'duration': format_info.get('duration')
                })
            
            if 'audio_stats' in diagnostic_results:
                results['analysis'].update(diagnostic_results['audio_stats'])
            
            if save_report:
                self._save_diagnostic_report(file_path, results)
            
            # Log key findings
            if results.get('errors'):
                logger.warning(f"Diagnostic errors found in {file_path}: {results['errors']}")
            if results.get('warnings'):
                logger.info(f"Diagnostic warnings for {file_path}: {results['warnings']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to diagnose {file_path}: {str(e)}"
            logger.error(error_msg)
            return {
                'file_path': file_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'errors': [error_msg],
                'warnings': [],
                'analysis': {},
                'success': False
            }
    
    def _save_diagnostic_report(self, file_path: str, results: Dict[str, Any]) -> str:
        """Save diagnostic report to JSON file"""
        try:
            # Generate report filename based on audio filename
            base_name = Path(file_path).stem
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            report_filename = f"diagnostic_{base_name}_{timestamp}.json"
            report_path = os.path.join(self.diagnostic_dir, report_filename)
            
            # Save the results
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Diagnostic report saved: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to save diagnostic report: {str(e)}")
            return ""
    
    def diagnose_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Run diagnostics on a batch of files
        
        Args:
            file_paths: List of file paths to diagnose
            
        Returns:
            List of diagnostic results for each file
        """
        results = []
        for file_path in file_paths:
            if os.path.exists(file_path) and file_path.lower().endswith('.wav'):
                result = self.diagnose_file(file_path)
                results.append(result)
            else:
                logger.warning(f"Skipping non-WAV or missing file: {file_path}")
        
        return results
    
    def get_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report from multiple diagnostic results
        
        Args:
            results: List of diagnostic results
            
        Returns:
            Summary report with aggregate statistics
        """
        summary = {
            'total_files': len(results),
            'successful_analyses': 0,
            'files_with_errors': 0,
            'files_with_warnings': 0,
            'common_issues': {},
            'quality_stats': {
                'avg_duration': 0,
                'avg_sample_rate': 0,
                'bit_depths': {},
                'channels': {}
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        total_duration = 0
        total_sample_rate = 0
        valid_files = 0
        
        for result in results:
            if result.get('success'):
                summary['successful_analyses'] += 1
                
                # Extract quality statistics
                analysis = result.get('analysis', {})
                if 'duration' in analysis:
                    total_duration += analysis['duration']
                    valid_files += 1
                
                if 'sample_rate' in analysis:
                    total_sample_rate += analysis['sample_rate']
                
                # Count bit depths and channels
                if 'bit_depth' in analysis:
                    bit_depth = str(analysis['bit_depth'])
                    summary['quality_stats']['bit_depths'][bit_depth] = summary['quality_stats']['bit_depths'].get(bit_depth, 0) + 1
                
                if 'channels' in analysis:
                    channels = str(analysis['channels'])
                    summary['quality_stats']['channels'][channels] = summary['quality_stats']['channels'].get(channels, 0) + 1
            
            # Count files with issues
            if result.get('errors'):
                summary['files_with_errors'] += 1
                for error in result['errors']:
                    summary['common_issues'][error] = summary['common_issues'].get(error, 0) + 1
            
            if result.get('warnings'):
                summary['files_with_warnings'] += 1
                for warning in result['warnings']:
                    summary['common_issues'][warning] = summary['common_issues'].get(warning, 0) + 1
        
        # Calculate averages
        if valid_files > 0:
            summary['quality_stats']['avg_duration'] = total_duration / valid_files
            summary['quality_stats']['avg_sample_rate'] = total_sample_rate / valid_files
        
        return summary
    
    def auto_diagnose_callback(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Callback function to be called after file generation
        
        Args:
            file_path: Path to the generated file
            
        Returns:
            Diagnostic results if auto_diagnose is enabled, None otherwise
        """
        if not self.auto_diagnose:
            return None
        
        return self.diagnose_file(file_path)
    
    def enable_auto_diagnose(self):
        """Enable automatic diagnosis of generated files"""
        self.auto_diagnose = True
        logger.info("Auto-diagnosis enabled")
    
    def disable_auto_diagnose(self):
        """Disable automatic diagnosis of generated files"""
        self.auto_diagnose = False
        logger.info("Auto-diagnosis disabled")
    
    def cleanup_old_reports(self, days: int = 30):
        """
        Clean up diagnostic reports older than specified days
        
        Args:
            days: Number of days to keep reports (default: 30)
        """
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            removed_count = 0
            
            for filename in os.listdir(self.diagnostic_dir):
                if filename.startswith('diagnostic_') and filename.endswith('.json'):
                    file_path = os.path.join(self.diagnostic_dir, filename)
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old diagnostic reports")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old reports: {str(e)}")

# Global instance for easy access
diagnostic_service = None

def get_diagnostic_service() -> DiagnosticService:
    """Get or create the global diagnostic service instance"""
    global diagnostic_service
    if diagnostic_service is None:
        # Check if diagnostics should be enabled by environment variable
        auto_diagnose = os.environ.get("ACE_AUTO_DIAGNOSE", "true").lower() == "true"
        diagnostic_service = DiagnosticService(auto_diagnose=auto_diagnose)
    return diagnostic_service

def initialize_diagnostic_service(diagnostic_dir: Optional[str] = None, auto_diagnose: bool = True) -> DiagnosticService:
    """Initialize the global diagnostic service with custom settings"""
    global diagnostic_service
    diagnostic_service = DiagnosticService(diagnostic_dir=diagnostic_dir, auto_diagnose=auto_diagnose)
    return diagnostic_service