#include <iostream>
#include <matcher.h>


int main(int argc, char** argv) {
	std::string target_cloud_path = "/home/rideflux/kict/rf_roadsign_detector/crop_test/gt_crop_pcd/00003.pcd";
	std::string source_cloud_path = "/home/rideflux/kict/rf_roadsign_detector/crop_test/infer_crop_pcd/00004.pcd";
	std::string result_pcd_path = "/home/rideflux/kict/rf_roadsign_detector/result.pcd";
	
	Matcher matcher(target_cloud_path,source_cloud_path);
	matcher.match(1);
	matcher.saveResultPCD(result_pcd_path);
	matcher.visualizeResult();
}
