from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from icecream import ic
import yt_dlp


video_url = "https://www.youtube.com/watch?v=sVcwVQRHIc8"

ydl_opts = {}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)  # Fetch metadata without downloading
    info_dict  # Print all metadata

    # Extract specific metadata
    title = info_dict.get("title")
    uploader = info_dict.get("uploader")
    upload_date = info_dict.get("upload_date")
    duration = info_dict.get("duration")
    views = info_dict.get("view_count")
    like_count = info_dict.get("like_count")

    ic(f"Title: {title}")
    ic(f"Uploader: {uploader}")
    ic(f"Upload Date: {upload_date}")
    ic(f"Duration (seconds): {duration}")
    ic(f"Views: {views}")
    ic(f"Likes: {like_count}")


