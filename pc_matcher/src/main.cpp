#include <iostream>
#include <toml.hpp>
#include <matcher.h>

int main(int argc, char** argv) {
  toml::table tbl;
  try{
    tbl = toml::parse_file(CONFIG_FILE_PATH);
  }catch (const toml::parse_error& err){
    std::cerr << "config 파일 경로 오류 \n" << err << std::endl;
    return 1;
  }

	std::string target_cloud_path = tbl["Paths"]["target_cloud_path"].value_or<std::string>("");
	std::string source_cloud_path = tbl["Paths"]["source_cloud_path"].value_or<std::string>("");
	std::string result_pcd_path   = tbl["Paths"]["result_pcd_path"].value_or<std::string>("");
	
	bool visualize_process = tbl["Settings"]["visualize_process"].value_or<bool>(false);
	bool visualize_result = tbl["Settings"]["visualize_result"].value_or<bool>(false);
	bool save_result_pcd = tbl["Settings"]["save_result_pcd"].value_or<bool>(false);

	Matcher matcher(target_cloud_path,source_cloud_path);
	matcher.match(visualize_process);
	if(save_result_pcd)
		matcher.saveResultPCD(result_pcd_path);
	if(visualize_result)
		matcher.visualizeResult();
}
