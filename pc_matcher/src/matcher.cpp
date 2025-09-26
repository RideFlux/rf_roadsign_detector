#include <matcher.h>

Matcher::Matcher(const std::string& target_path, const std::string& source_path):
  target_cloud_(new pcl::PointCloud<Point_T>()),
  source_cloud_(new pcl::PointCloud<Point_T>()),
  updated_cloud_(new pcl::PointCloud<Point_T>()),
  kdtree_(new pcl::search::KdTree<Point_T>())
{
   if (pcl::io::loadPCDFile(target_path, *target_cloud_)) {
    std::cerr << "failed to open " << target_path << std::endl;
  }
  if (pcl::io::loadPCDFile(source_path, *source_cloud_)) {
    std::cerr << "failed to open " << source_path << std::endl;
  } 

  if(!target_cloud_ || target_cloud_->points.empty()){
    std::cerr << "target cloud is empty " << std::endl;
  }

  if(!source_cloud_ || source_cloud_->points.empty()){
    std::cerr << "source cloud is empty " << std::endl;
  }
}

void Matcher::match(bool visualize_process){
  
  Eigen::Affine3f current_transform = Eigen::Affine3f::Identity();
  pcl::PointCloud<Point_T>::Ptr updated_cloud(new pcl::PointCloud<Point_T>());
  optimizer_.setTargetCloud(target_cloud_);

  for(int iter=0; iter<MAX_ITER_NUM; iter++){
    pcl::transformPointCloud(*source_cloud_, *updated_cloud, current_transform);

    findCorrespondences(updated_cloud,target_cloud_);
    optimizer_.setSourceCloud(updated_cloud);
    optimizer_.setCorrespondence(correspondence_vec_);
    optimizer_.optimize();
    auto update = optimizer_.getResult();
    optimizer_.clear();

    if(visualize_process){
      static pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(" "));
      viewer->addCoordinateSystem(1.0);
      viewer->initCameraParameters();
      viewer->setCameraPosition(source_cloud_->points[0].x + 5.0 ,source_cloud_->points[0].y - 5.0, 0, 
                                target_cloud_->points[0].x + 5.0 ,target_cloud_->points[0].y - 5.0, 0, 
                                0,0,1);
      visualizeRegistration(viewer, target_cloud_,source_cloud_,updated_cloud,correspondence_vec_);
    }

    current_transform = update * current_transform;

    auto delta_t = update.translation().norm();
    auto delta_R = Eigen::AngleAxisf(update.rotation()).angle();

    if(delta_R < CONVERGE_THRESHOLD_ROTATION && delta_t < CONVERGE_THRESHOLD_TRANSLATION) 
      break;
  }
  // std::cout << "result : " << current_transform.matrix() << std::endl;
  *updated_cloud_ = *updated_cloud;
}

void Matcher::findCorrespondences(pcl::PointCloud<Point_T>::Ptr in_cloud, pcl::PointCloud<Point_T>::Ptr ref_cloud){
  correspondence_vec_.clear();
  correspondence_vec_.reserve(in_cloud->points.size());

  kdtree_->setInputCloud(ref_cloud);

  std::vector<int> idx_vec;
  std::vector<float> dist_vec;

  for(std::size_t in_idx=0; in_idx<in_cloud->points.size(); in_idx++){
    const auto& query = in_cloud->points[in_idx];
    kdtree_->nearestKSearch(query, 1, idx_vec,dist_vec);
    const int ref_idx = idx_vec[0];
    // Point_T closest_target_point = target_cloud_->points[ref_idx];

    correspondence_vec_.emplace_back(Optimizer::Correspondence(in_idx,ref_idx));
  }
}

void Matcher::saveResultPCD(const std::string& path){

  if (pcl::io::savePCDFileASCII(path, *updated_cloud_) == -1) {
    std::cerr << "cannot save result pcd " << path << std::endl;
  }
}
void Matcher::visualizeResult(){
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Result"));
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  viewer->setCameraPosition(source_cloud_->points[0].x + 5.0 ,source_cloud_->points[0].y - 5.0, 0, 
                                target_cloud_->points[0].x + 5.0 ,target_cloud_->points[0].y - 5.0, 0, 
                                0,0,1);
  visualizeRegistration(viewer,target_cloud_,source_cloud_,updated_cloud_);
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}

void Matcher::visualizeRegistration(
  const pcl::visualization::PCLVisualizer::Ptr& viewer,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& result,
  const std::vector<Optimizer::Correspondence>& correspondence_vec) {
  
  viewer->removeAllPointClouds();
  viewer->removeAllShapes();
  viewer->setBackgroundColor(0.1, 0.1, 0.1); // 배경 어둡게

  // Target 포인트 클라우드 (흰색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target, 255, 255, 255);
  viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "target cloud");
  viewer->addText("WHITE : Target Cloud", 10, 50, 16, 1.0, 1.0, 1.0, "target_text");

  // Source 포인트 클라우드 (빨간색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source, 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(source, source_color, "source cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "source cloud");
  viewer->addText("   RED   : Source Cloud", 10, 70, 16, 1.0, 0.0, 0.0, "source_text");

  // 정합된  포인트 클라우드 (녹색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_color(result, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(result, aligned_color, "aligned cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "aligned cloud");
  viewer->addText("GREEN : Result Cloud", 10, 90, 16, 0.0, 1.0, 0.0, "result_text");

  if(!correspondence_vec.empty()){
    for(std::size_t i=0; i<correspondence_vec.size(); i++){
      const auto& corr = correspondence_vec[i];
      const auto& src = result->points[corr.in_idx];
      const auto& tgt = target->points[corr.ref_idx];

      std::string line_id = "line" + std::to_string(i);
      viewer->addLine<Point_T>(src, tgt, 255, 255, 0, line_id);
    }
  }

  viewer->spinOnce(100);
}