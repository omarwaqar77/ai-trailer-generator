import pandas as pd
from moviepy import VideoFileClip, concatenate_videoclips
import os

def generate_trailer(features_csv, data_dir, output_file):
    print(f"Loading features from {features_csv}...")
    df = pd.read_csv(features_csv)
    
    # Filter only High Impact clips
    high_impact = df[df['label'] == 1].copy()
    
    if len(high_impact) < 5:
        print("Warning: Found fewer than 5 high-impact clips. Padding with top low-impact clips.")
        low_impact = df[df['label'] == -1].sort_values(by='heuristic_score', ascending=False)
        needed = 5 - len(high_impact)
        high_impact = pd.concat([high_impact, low_impact.head(needed)])
        
    # Select the Top 5 best clips based on heuristic score
    top_5 = high_impact.sort_values(by='heuristic_score', ascending=False).head(5)
    
    # Narrative Flow Sorting:
    # 1. We want the lowest motion clip first (build suspense)
    # 2. Then the rest sorted by increasing object density (climax hint)
    
    # Sort by motion to find the start
    top_5_sorted_motion = top_5.sort_values(by='motion_intensity')
    start_clip = top_5_sorted_motion.iloc[[0]]
    
    # The remaining 4 clips sorted by object density to build up to a climax
    remaining_clips = top_5_sorted_motion.iloc[1:].sort_values(by='object_density')
    
    final_sequence = pd.concat([start_clip, remaining_clips])
    
    print("\nSelected Clips for Trailer (in order of appearance):")
    for idx, row in enumerate(final_sequence.itertuples()):
        print(f"{idx+1}. {row.clip_name} (Score: {row.heuristic_score:.2f}, Motion: {row.motion_intensity:.2f}, Objects: {row.object_density:.2f})")
        
    # Concatenate using MoviePy
    print("\nMerging clips into trailer...")
    clips = []
    for clip_name in final_sequence['clip_name']:
        clip_path = os.path.join(data_dir, clip_name)
        # Load clip
        video = VideoFileClip(clip_path)
        clips.append(video)
        
    # Final trailer
    final_trailer = concatenate_videoclips(clips, method="compose")
    
    # Write to file
    final_trailer.write_videofile(output_file, codec="libx264", audio_codec="aac", fps=30)
    print(f"\nRaw trailer generated successfully: {output_file}")
    
    # Close clips to free memory
    for c in clips:
        c.close()

if __name__ == '__main__':
    features_csv = r"d:\Semester 4\AI LAB\OEL\labeled_features.csv"
    data_dir = r"d:\Semester 4\AI LAB\OEL\data"
    output_file = r"d:\Semester 4\AI LAB\OEL\trailer_raw.mp4"
    
    generate_trailer(features_csv, data_dir, output_file)
