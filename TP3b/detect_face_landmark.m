function detections = detect_face_landmark(imgfilename)
%imgfilename can be a single img filename or a cell of multiple filename
%it returns one detections struct for each img filename

if iscell(imgfilename)
    filenames = sprintf('%s ', imgfilename{:}); %only works if the utrecht is in the matlab path
    system(['face_landmark_detection.exe shape_predictor_68_face_landmarks.dat ' filenames]);
    detections(length(imgfilename)) = struct('rects',[],'features',[]);
    for i=1:length(imgfilename)
        filename = imgfilename{i}; filename = filename(1:end-4);
        detections(i) = parse_landmarkfile(filename);
        delete(filename);
    end
else
    system(['face_landmark_detection.exe shape_predictor_68_face_landmarks.dat ' imgfilename]);
    filename = imgfilename(1:end-4);
    detections = parse_landmarkfile(filename);
    delete(filename);
end

function detections = parse_landmarkfile(landmarkfilename)

data = csvread(landmarkfilename);
rects = data(1:2:end,1:4);
tmp = data(2:2:end,:);
features = zeros(2,size(tmp,2)/2,size(tmp,1));
for i=1:size(tmp,1)
    features(1,:,i) = tmp(i,1:2:end);
    features(2,:,i) = tmp(i,2:2:end);
end

detections = struct('rects',rects,'features',features);