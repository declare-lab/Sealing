import json

def load_video_paths(args):
    """load a list of (path, image_id,tuples"""
    video_paths = []
    video_ids = []
    with open(args.annotation_file, 'r') as anno_file:
        instances = json.load(anno_file)
    [video_ids.append(int(instance['id'])) for instance in instances]
    video_ids = set(video_ids)
    for video_id in video_ids:
        video_paths.append((args.video_dir + '{}.mp4'.format(str(video_id)), video_id))
    return video_paths