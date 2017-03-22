function h = show_landmark(img,rects,points)

h = gcf;
imshow(img); hold on;
for i=1:size(rects,1)
    rectangle('Position',[rects(i,1) rects(i,2) rects(i,3)-rects(i,1) rects(i,4)-rects(i,2)],'edgecolor','r');
    plot(points(1,:,i),points(2,:,i),'.b');
end
hold off;